"""Helpers for the RestoreFormer++ worker (ONNX, no torch dependency)."""

import base64
import binascii
import io
import os
from typing import Any

import numpy as np
from PIL import Image

MODELS_ROOT = os.environ.get("MODELS_ROOT", "/runpod-volume")
MAX_INPUT_PIXELS = int(os.environ.get("MAX_INPUT_PIXELS", 12_000_000))


class BadInput(ValueError):
    pass


def decode_image(payload: dict[str, Any]) -> Image.Image:
    raw = payload.get("image") or payload.get("image_base64")
    if not isinstance(raw, str) or not raw:
        raise BadInput("No image supplied.")
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise BadInput("The image was not valid base64.") from exc
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise BadInput("The image could not be decoded.") from exc
    if image.width * image.height > MAX_INPUT_PIXELS:
        raise BadInput(f"Image is too large ({image.width}x{image.height}).")
    return image


def encode_image(image: Image.Image, *, keep_alpha: bool = False) -> str:
    buffer = io.BytesIO()
    image.convert("RGBA" if keep_alpha else "RGB").save(
        buffer, format="PNG", optimize=False, compress_level=3
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")
