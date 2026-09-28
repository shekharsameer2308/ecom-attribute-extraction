"""
Image preprocessing module.
Handles quality checks, rotation, resizing, and tiling before sending to the LLM.
"""

import os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageFilter, ImageStat


# ---------------------------------------------------------------------------
# Image quality analysis
# ---------------------------------------------------------------------------

def analyze_image_quality(img: Image.Image) -> Dict[str, any]:
    """
    Runs heuristic checks on an image and returns quality flags.
    These flags are passed alongside the LLM response so the pipeline
    can decide whether to trust the extraction or request a better photo.
    """
    flags = []
    width, height = img.size

    # 1. Resolution check
    if width < 300 or height < 300:
        flags.append("low_resolution")

    # 2. Blur detection (Laplacian variance approximation via PIL)
    #    Convert to greyscale, apply edge-detect filter, measure variance.
    grey = img.convert("L")
    edges = grey.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    edge_variance = stat.var[0]
    if edge_variance < 50:          # threshold tuned on sample images
        flags.append("blurry")

    # 3. Brightness / glare check
    brightness_stat = ImageStat.Stat(grey)
    mean_brightness = brightness_stat.mean[0]
    if mean_brightness > 240:
        flags.append("glare")
    elif mean_brightness < 30:
        flags.append("too_dark")

    # 4. Aspect ratio — extremely tall/wide images may be cropped
    aspect = max(width, height) / max(min(width, height), 1)
    if aspect > 4.0:
        flags.append("cropped")

    usable = "blurry" not in flags and "low_resolution" not in flags
    return {
        "width": width,
        "height": height,
        "flags": flags,
        "usable": usable,
        "edge_variance": round(edge_variance, 2),
        "mean_brightness": round(mean_brightness, 2),
    }


# ---------------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------------

def preprocess_image(
    image_path: str,
    max_dimension: int = 2048,
    output_dir: Optional[str] = None,
) -> str:
    """
    Prepares a single image for the LLM:
      1. Auto-rotate using EXIF orientation
      2. Resize if any dimension exceeds max_dimension
      3. Convert RGBA → RGB (Gemini requires JPEG/PNG without alpha issues)
      4. Save the processed image and return its path

    If output_dir is None the processed image is saved next to the original
    with a '_processed' suffix.
    """
    img = Image.open(image_path)

    # Auto-rotate based on EXIF
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # no EXIF data, continue

    # Resize keeping aspect ratio
    width, height = img.size
    if max(width, height) > max_dimension:
        ratio = max_dimension / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert RGBA → RGB
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Save processed image
    base, ext = os.path.splitext(os.path.basename(image_path))
    out_name = f"{base}_processed.jpg"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, out_name)
    else:
        out_path = os.path.join(os.path.dirname(image_path), out_name)

    img.save(out_path, "JPEG", quality=90)
    return out_path


def tile_large_image(
    image_path: str,
    tile_size: int = 1024,
    overlap: int = 128,
    output_dir: Optional[str] = None,
) -> List[str]:
    """
    Splits a very large or panoramic image into overlapping tiles.
    Useful when packaging wraps around a cylinder and a single shot
    captures the full surface (very wide images).
    Returns a list of tile image paths.
    """
    img = Image.open(image_path)
    width, height = img.size
    tiles = []

    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    step = tile_size - overlap
    tile_idx = 0

    y = 0
    while y < height:
        x = 0
        while x < width:
            box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            tile = img.crop(box)
            tile_path = os.path.join(output_dir, f"{base}_tile_{tile_idx}.jpg")
            tile.save(tile_path, "JPEG", quality=90)
            tiles.append(tile_path)
            tile_idx += 1
            x += step
        y += step

    return tiles
