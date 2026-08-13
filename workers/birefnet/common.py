"""Helpers shared by every Lumina worker.

Kept as one small file that gets copied into each image rather than a package,
so a worker can be built and reasoned about on its own.
"""

import base64
import binascii
import io
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

# Runpod mounts the network volume here for serverless workers.
MODELS_ROOT = os.environ.get("MODELS_ROOT", "/runpod-volume")

# Guard rail: the gateway already caps uploads, but a worker should never be
# talked into allocating unbounded VRAM by a malformed request.
MAX_INPUT_PIXELS = int(os.environ.get("MAX_INPUT_PIXELS", 12_000_000))


class BadInput(ValueError):
    """Raised for anything wrong with the request payload."""


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_image(payload: dict[str, Any]) -> Image.Image:
    """Pull the base64 image out of a Runpod `input` object."""
    raw = payload.get("image") or payload.get("image_base64")
    if not isinstance(raw, str) or not raw:
        raise BadInput("No image supplied.")

    # Accept data URLs as well as bare base64.
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")

    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise BadInput("The image was not valid base64.") from exc

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow raises a wide range here
        raise BadInput("The image could not be decoded.") from exc

    if image.width * image.height > MAX_INPUT_PIXELS:
        raise BadInput(
            f"Image is too large ({image.width}x{image.height}). "
            f"The limit is {MAX_INPUT_PIXELS:,} pixels."
        )

    return image


def encode_image(
    image: Image.Image, *, keep_alpha: bool = False, max_size_mb: int = 8
) -> str:
    """Encode a result as base64, PNG only.

    Runpod's job-done endpoint rejects payloads over ~10MB. JPEG is not an
    option here when `keep_alpha` is set — it has no alpha channel, and a
    cut-out without one is not a cut-out — so oversized results are instead
    brought under the limit by raising PNG compression first and, if that
    still is not enough, downscaling before re-encoding. Detail lost this way
    only matters on cut-outs from very large source photos, which is a far
    better trade than the job failing outright.
    """
    rgba = image.convert("RGBA" if keep_alpha else "RGB")
    max_bytes = max_size_mb * 1024 * 1024

    buffer = io.BytesIO()
    rgba.save(buffer, format="PNG", optimize=False, compress_level=3)
    if buffer.tell() <= max_bytes:
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    buffer = io.BytesIO()
    rgba.save(buffer, format="PNG", optimize=True, compress_level=9)
    if buffer.tell() <= max_bytes:
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    # Still too large: scale down and re-encode. Halving repeatedly rather than
    # computing an exact target — alpha-matted edges compress unpredictably, so
    # an exact byte target is not reliable to hit in one step anyway.
    current = rgba
    for _ in range(4):
        current = current.resize(
            (max(current.width // 2, 1), max(current.height // 2, 1)),
            Image.LANCZOS,
        )
        buffer = io.BytesIO()
        current.save(buffer, format="PNG", optimize=True, compress_level=9)
        if buffer.tell() <= max_bytes:
            break

    return base64.b64encode(buffer.getvalue()).decode("ascii")


def to_tensor(image: Image.Image) -> torch.Tensor:
    """HWC uint8 PIL → NCHW float32 tensor in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def to_image(tensor: torch.Tensor) -> Image.Image:
    """NCHW float tensor in [0, 1] → PIL."""
    array = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def free_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
