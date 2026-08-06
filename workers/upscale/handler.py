"""Runpod worker for every upscaling model.

One image serves three of Lumina's tools, because `spandrel` recognises all
three architectures from their weight files:

    nomos  → 4xNomos2_hq_dat2.safetensors     (DAT2)   Photo Enhancer, Artwork
    faceup → 4xFaceUpSharpDAT.safetensors     (DAT)    Portrait Enhance
    anime  → RealESRGAN_x4plus_anime_6B.pth   (RRDBNet) Anime Enhancer

Deploy it once and point three endpoint ids at it, or three times — the gateway
does not care. Which weights to use comes from `input.model`.

Contract:
    input : { "image": "<base64>", "model": "nomos|faceup|anime", "scale": 2|4|8|16, "tile": 512 }
    output: { "image": "<base64 png>" }
"""

import gc
import os
from typing import Any

import runpod
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

from common import (
    MODELS_ROOT,
    BadInput,
    decode_image,
    device,
    encode_image,
    free_vram,
    to_image,
    to_tensor,
)

# Model key → path on the network volume.
WEIGHTS = {
    "nomos": f"{MODELS_ROOT}/upscale/4xNomos2_hq_dat2.safetensors",
    "faceup": f"{MODELS_ROOT}/faceupscale/4xFaceUpSharpDAT.safetensors",
    "anime": f"{MODELS_ROOT}/realesrgan/RealESRGAN_x4plus_anime_6B.pth",
}

SUPPORTED_SCALES = (2, 4, 8, 16)

# Loaded models stay resident between jobs on a warm worker — reloading a DAT
# checkpoint per request would dominate the actual inference time.
_cache: dict[str, ImageModelDescriptor] = {}


def load_model(key: str) -> ImageModelDescriptor:
    if key in _cache:
        return _cache[key]

    path = WEIGHTS.get(key)
    if path is None:
        raise BadInput(f"Unknown model '{key}'.")
    if not os.path.exists(path):
        # Almost always a missing or wrongly-mounted network volume.
        raise RuntimeError(f"Weights not found at {path}.")

    model = ModelLoader().load_from_file(path)
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError(f"{key} is not an image-to-image model.")

    model.to(device()).eval()

    # Only one architecture stays in VRAM at a time: three DAT models resident
    # together is a needless few gigabytes on a worker that handles one job.
    for other in list(_cache):
        if other != key:
            _cache.pop(other)
    gc.collect()
    free_vram()

    _cache[key] = model
    return model


@torch.inference_mode()
def upscale_tiled(
    model: ImageModelDescriptor,
    image: Image.Image,
    *,
    tile: int,
    overlap: int = 32,
) -> Image.Image:
    """Run the model in overlapping tiles.

    A 4x DAT pass on a 12MP photo will not fit in 24GB in one go. Tiles keep
    memory flat regardless of input size, and the overlap is blended away so
    there are no seams.
    """
    dev = device()
    scale = model.scale
    source = to_tensor(image).to(dev)
    _, _, height, width = source.shape

    # Small enough to do in one pass.
    if tile <= 0 or (height <= tile and width <= tile):
        return to_image(model(source).float())

    output = torch.zeros(
        (1, 3, height * scale, width * scale), dtype=torch.float32, device=dev
    )
    weights = torch.zeros_like(output)

    step = max(tile - overlap, 1)
    for top in range(0, height, step):
        for left in range(0, width, step):
            bottom = min(top + tile, height)
            right = min(left + tile, width)
            # Pull the window back so edge tiles stay full-sized.
            top_in = max(bottom - tile, 0)
            left_in = max(right - tile, 0)

            patch = source[:, :, top_in:bottom, left_in:right]
            result = model(patch).float()

            out_top, out_left = top_in * scale, left_in * scale
            out_bottom, out_right = bottom * scale, right * scale

            # Feathered edges, so overlapping tiles average instead of stepping.
            mask = torch.ones_like(result)
            feather = overlap * scale
            if feather > 0:
                ramp = torch.linspace(0, 1, feather, device=dev)
                if out_top > 0:
                    mask[:, :, :feather, :] *= ramp.view(1, 1, -1, 1)
                if out_left > 0:
                    mask[:, :, :, :feather] *= ramp.view(1, 1, 1, -1)
                if out_bottom < height * scale:
                    mask[:, :, -feather:, :] *= ramp.flip(0).view(1, 1, -1, 1)
                if out_right < width * scale:
                    mask[:, :, :, -feather:] *= ramp.flip(0).view(1, 1, 1, -1)

            output[:, :, out_top:out_bottom, out_left:out_right] += result * mask
            weights[:, :, out_top:out_bottom, out_left:out_right] += mask

            del patch, result, mask

    output /= weights.clamp(min=1e-8)
    image_out = to_image(output)

    del source, output, weights
    free_vram()
    return image_out


def apply_scale(
    model: ImageModelDescriptor,
    image: Image.Image,
    *,
    target: int,
    tile: int,
) -> Image.Image:
    """Reach the requested factor using a fixed-4x model.

    The models are all 4x. Anything else is composed from that:
        2x  → one pass, then Lanczos down (sharper than a plain 2x resize)
        4x  → one pass
        8x  → two passes (16x), then Lanczos down
        16x → two passes
    """
    native = model.scale
    passes = 1 if target <= native else 2

    result = image
    for _ in range(passes):
        result = upscale_tiled(model, result, tile=tile)

    reached = native**passes
    if reached != target:
        width = round(image.width * target)
        height = round(image.height * target)
        result = result.resize((width, height), Image.LANCZOS)

    return result


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input") or {}

    try:
        image = decode_image(payload)

        key = str(payload.get("model", "nomos")).lower()
        scale = int(payload.get("scale", 4))
        if scale not in SUPPORTED_SCALES:
            raise BadInput(f"Unsupported scale {scale}.")

        tile = int(payload.get("tile", 512))

        model = load_model(key)
        result = apply_scale(model, image, target=scale, tile=tile)

        return {"image": encode_image(result)}

    except BadInput as exc:
        # A client mistake, not a server fault — do not retry this job.
        return {"error": str(exc)}
    except torch.cuda.OutOfMemoryError:
        free_vram()
        return {
            "error": "Ran out of GPU memory. Try a smaller image or a lower "
            "scale."
        }
    except Exception as exc:  # noqa: BLE001 - surface anything else as a job error
        free_vram()
        return {"error": f"Upscale failed: {exc}"}


runpod.serverless.start({"handler": handler})
