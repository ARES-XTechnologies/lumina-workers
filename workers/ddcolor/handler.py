"""Runpod worker for DDColor (Photo Colourisation).

The trick that makes colourisation look right: the model only predicts colour,
never brightness. It runs at 512x512, and only the two chroma channels of its
output are kept — the luminance comes from the original at full resolution.

That means a 4000px photo keeps every bit of its detail and just gains colour,
instead of coming back as a soft 512px upscale.

Contract:
    input : { "image": "<base64>" }
    output: { "image": "<base64 png>" }
"""

import os
from typing import Any

import cv2
import numpy as np
import runpod
import torch
from PIL import Image

from common import BadInput, MODELS_ROOT, decode_image, device, encode_image, free_vram
from ddcolor_arch import DDColor

WEIGHTS = f"{MODELS_ROOT}/ddcolor/pytorch_model.bin"

INPUT_SIZE = 512

_model: DDColor | None = None


def load_model() -> DDColor:
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(WEIGHTS):
        raise RuntimeError(f"Weights not found at {WEIGHTS}.")

    model = DDColor(
        encoder_name="convnext-l",
        decoder_name="MultiScaleColorDecoder",
        input_size=(INPUT_SIZE, INPUT_SIZE),
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )

    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    state = state.get("params", state)
    model.load_state_dict(state, strict=False)

    model.to(device()).eval()
    _model = model
    return _model


@torch.inference_mode()
def colourise(image: Image.Image) -> Image.Image:
    model = load_model()
    dev = device()

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]

    # Original luminance, kept at full resolution.
    orig_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[:, :, :1]

    # The model wants a greyscale image expressed in Lab, at 512x512.
    small = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    small_l = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)[:, :, :1]
    grey_lab = np.concatenate(
        [small_l, np.zeros_like(small_l), np.zeros_like(small_l)], axis=-1
    )
    grey_rgb = cv2.cvtColor(grey_lab, cv2.COLOR_Lab2RGB)

    tensor = torch.from_numpy(grey_rgb.transpose(2, 0, 1))[None, ...].to(dev)

    output_ab = model(tensor).cpu().numpy()[0].transpose(1, 2, 0)

    # Chroma back up to full size, luminance untouched.
    output_ab = cv2.resize(output_ab, (width, height), interpolation=cv2.INTER_LINEAR)
    result_lab = np.concatenate([orig_l, output_ab], axis=-1)
    result = cv2.cvtColor(result_lab, cv2.COLOR_Lab2RGB)

    free_vram()
    return Image.fromarray((np.clip(result, 0, 1) * 255.0).round().astype(np.uint8))


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input") or {}

    try:
        image = decode_image(payload)
        return {"image": encode_image(colourise(image))}

    except BadInput as exc:
        return {"error": str(exc)}
    except torch.cuda.OutOfMemoryError:
        free_vram()
        return {"error": "Ran out of GPU memory. Try a smaller image."}
    except Exception as exc:  # noqa: BLE001
        free_vram()
        return {"error": f"Colourisation failed: {exc}"}


runpod.serverless.start({"handler": handler})
