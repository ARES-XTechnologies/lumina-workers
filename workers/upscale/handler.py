"""Runpod worker for Auto Enhance.

`spandrel` loads any of these from the weight file alone, so one image can serve
several models and the gateway picks with `input.model`:

    x4plus → RealESRGAN_x4plus.pth         (RRDBNet)  Auto Enhance
    nomos  → 4xNomos2_hq_dat2.safetensors  (DAT2)     spare
    faceup → 4xFaceUpSharpDAT.safetensors  (DAT)      spare

Only x4plus is wired up today. The two spares stay listed because adding a
content router later should be a gateway change, not a worker rebuild.

The anime variant is deliberately gone. It over-sharpens photographs, and with
one Auto Enhance path there is nothing to route a drawing to — an option that
can only be reached by mistake is worse than no option.

Contract:
    input : { "image": "<base64>", "model": "x4plus|nomos|faceup", "scale": 2|4, "tile": 512 }
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
    # Auto Enhance. x4plus is trained on real degraded photographs and holds up
    # across everything users actually upload.
    "x4plus": f"{MODELS_ROOT}/realesrgan/RealESRGAN_x4plus.pth",
    # Kept available for a future content router; nothing calls them today.
    "nomos": f"{MODELS_ROOT}/upscale/4xNomos2_hq_dat2.safetensors",
    "faceup": f"{MODELS_ROOT}/faceupscale/4xFaceUpSharpDAT.safetensors",
}

SUPPORTED_SCALES = (2, 4)

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

    The model is 4x natively. 2x is that pass scaled back down with Lanczos,
    which is sharper than asking for 2x directly — the detail is recovered at
    full strength and then resampled, rather than never being recovered.
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

        key = str(payload.get("model", "x4plus")).lower()
        scale = int(payload.get("scale", 4))
        if scale not in SUPPORTED_SCALES:
            raise BadInput(f"Unsupported scale {scale}.")

        tile = int(payload.get("tile", 512))

        model = load_model(key)
        result = apply_scale(model, image, target=scale, tile=tile)

        return {"image": encode_image(_add_grain(result), max_size_mb=8)}

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


# Pre-load x4plus at startup so FlashBoot can pause with it in VRAM.
# Every other model (nomos, faceup) stays lazy — they are spares.
try:
    _preloaded = load_model("x4plus")
    print("[lumina] x4plus preloaded", flush=True)
except Exception as exc:
    print(f"[lumina] preload failed: {exc}", flush=True)

def _add_grain(image: Image.Image) -> Image.Image:
    """Add subtle luminance-based film grain to break up the plastic look.

    Real film grain is:
    1. Luminance-dependent — more grain in midtones, less in shadows/highlights
    2. Slightly colour-tinted — not pure grey noise
    3. Spatially correlated — not pure per-pixel random

    A pure random noise overlay on every pixel looks digital and harsh.
    This version applies stronger grain where the image is mid-tone and
    almost none in very dark or very bright areas — exactly like real film.
    """
    arr = np.array(image).astype(np.float32)

    # Luminance mask: grain is strongest in midtones (around 128)
    # and fades toward 0 in shadows and highlights.
    # Formula: 1 - |L/128 - 1|^1.5 gives a smooth bell curve.
    lum = arr.mean(axis=2, keepdims=True) / 255.0
    mask = 1.0 - np.abs(lum - 0.5) ** 1.2
    mask = np.clip(mask, 0.05, 1.0)

    # Grain strength: 4.5 gives a natural film look at normal viewing distance
    # and visible texture when zoomed — not so strong it looks artificially noisy.
    strength = 4.5

    # Generate spatially correlated noise by creating at half resolution
    # and upsampling — this gives the slight clustering that real grain has
    # rather than pure per-pixel randomness.
    h, w = arr.shape[:2]
    small_h, small_w = max(h // 2, 1), max(w // 2, 1)
    small_noise = np.random.normal(0, 1, (small_h, small_w, 1)).astype(np.float32)

    # Upsample with bilinear — smooth clusters, not blocky pixels
    from PIL import Image as _Image
    noise_img = _Image.fromarray(
        np.clip(small_noise[:, :, 0] * 127 + 128, 0, 255).astype(np.uint8), mode="L"
    )
    noise_img = noise_img.resize((w, h), _Image.BILINEAR)
    noise = (np.array(noise_img).astype(np.float32) - 128.0)[..., None]

    # Apply: luminance-weighted noise, broadcast across RGB channels
    grained = arr + noise * mask * strength

    return _Image.fromarray(np.clip(grained, 0, 255).astype(np.uint8))


runpod.serverless.start({"handler": handler})
