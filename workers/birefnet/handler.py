"""Runpod worker for BiRefNet (Background Removal).

BiRefNet predicts a soft alpha matte rather than a hard mask, which is what
makes hair and fur look right instead of cut out with scissors. The matte comes
back at 1024x1024 and is resized to the source resolution before compositing.

Contract:
    input : { "image": "<base64>", "output_format": "png", "alpha_matting": true }
    output: { "image": "<base64 png with alpha>" }
"""

import os
from typing import Any

import numpy as np
import runpod
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from common import BadInput, MODELS_ROOT, decode_image, device, encode_image, free_vram

MODEL_DIR = os.environ.get("MODELS_ROOT", "/app/weights") + "/birefnet"

# The resolution BiRefNet was trained at. Feeding it anything else costs
# accuracy at the edges, so the image is resized in and the matte resized out.
INPUT_SIZE = 1024

_model = None

_preprocess = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model():
    """Load once per worker; the checkpoint is 424MB."""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        raise RuntimeError(f"Weights not found in {MODEL_DIR}.")

    # trust_remote_code: BiRefNet ships its architecture as birefnet.py next to
    # the weights, so the config files must be on the volume too — see
    # workers/README.md.
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )

    dev = device()
    if dev.type == "cuda":
        # Half precision halves VRAM and is visually indistinguishable for a
        # matte.
        model = model.half()
        torch.set_float32_matmul_precision("high")

    model.to(dev).eval()
    _model = model
    return _model


@torch.inference_mode()
def predict_matte(image: Image.Image) -> Image.Image:
    """Return a single-channel alpha matte at the source resolution."""
    model = load_model()
    dev = device()

    tensor = _preprocess(image.convert("RGB")).unsqueeze(0).to(dev)
    if dev.type == "cuda":
        tensor = tensor.half()

    # BiRefNet returns a list of supervision outputs; the last is the finest.
    prediction = model(tensor)[-1].sigmoid().float().cpu()

    matte = prediction[0].squeeze()
    matte = (matte - matte.min()) / (matte.max() - matte.min() + 1e-8)

    alpha = Image.fromarray(
        (matte.numpy() * 255.0).round().astype(np.uint8), mode="L"
    )
    free_vram()
    return alpha.resize(image.size, Image.LANCZOS)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input") or {}

    try:
        image = decode_image(payload).convert("RGB")
        alpha = predict_matte(image)

        cutout = image.convert("RGBA")
        cutout.putalpha(alpha)

        # Always PNG: a cut-out without an alpha channel is not a cut-out.
        return {"image": encode_image(cutout, keep_alpha=True)}

    except BadInput as exc:
        return {"error": str(exc)}
    except torch.cuda.OutOfMemoryError:
        free_vram()
        return {"error": "Ran out of GPU memory. Try a smaller image."}
    except Exception as exc:  # noqa: BLE001
        free_vram()
        return {"error": f"Background removal failed: {exc}"}


# Preload at startup — with the model in VRAM, FlashBoot pauses here
# and resumes with zero load time on the next request.
try:
    load_model()
    print("[lumina] BiRefNet preloaded", flush=True)
except Exception as exc:
    print(f"[lumina] preload failed: {exc}", flush=True)

runpod.serverless.start({"handler": handler})
