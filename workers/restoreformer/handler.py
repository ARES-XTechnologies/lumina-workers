"""Runpod worker for RestoreFormer++ (Face Restoration).

Restoring a face is three steps, and skipping any of them is why naive
implementations look worse than the demos:

    1. find the faces, and the transform that squares each one up
    2. restore each aligned 512x512 crop
    3. warp the results back and blend them into the original

Doing (2) on the whole photo instead — the obvious shortcut — means the model
sees a face 80 pixels tall and returns mush.

Face detection is OpenCV's Haar cascade, bundled inside the opencv package
itself — nothing to download, nothing that can go missing from a Docker build.
An earlier version used the DNN SSD detector, which needed two extra weight
files baked into the image; when that download silently failed the detector
ended up half-initialised and the job crashed with an OpenCV assertion instead
of a clear error. The cascade removes that whole failure mode.

Contract:
    input : { "image": "<base64>", "fidelity": 0.5, "only_center_face": false }
    output: { "image": "<base64 png>" }
"""

import os
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import runpod
from PIL import Image

from common import MODELS_ROOT, BadInput, decode_image, encode_image

MODEL_PATH = f"{MODELS_ROOT}/restoreformer/RestoreFormerPlusPlus.onnx"
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

FACE_SIZE = 512

# Faces smaller than this fraction of the frame's shorter side are treated as
# incidental rather than the subject — restoring them would upscale noise.
MIN_FACE_FRACTION = 0.04

_session: ort.InferenceSession | None = None
_cascade: cv2.CascadeClassifier | None = None


def session() -> ort.InferenceSession:
    """Lazily create the ONNX session and keep it for the worker's lifetime."""
    global _session
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Weights not found at {MODEL_PATH}.")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        _session = ort.InferenceSession(
            MODEL_PATH, sess_options=options, providers=providers
        )
    return _session


def detector() -> cv2.CascadeClassifier | None:
    """Haar cascade face detector. None only if OpenCV's own data is missing,
    which would mean a broken install rather than anything we can fix here."""
    global _cascade
    if _cascade is None:
        cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        _cascade = None if cascade.empty() else cascade
    return _cascade


def find_faces(
    bgr: np.ndarray, *, only_center: bool
) -> list[tuple[int, int, int, int]]:
    """Boxes for each detected face, largest first, padded for hairline/chin."""
    height, width = bgr.shape[:2]
    cascade = detector()

    if cascade is None:
        # No detector available: treat the whole frame as one face. Correct
        # for already-cropped portraits, which is the common case here.
        return [(0, 0, width, height)]

    grey = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    min_side = int(min(width, height) * MIN_FACE_FRACTION)

    faces = cascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(max(min_side, 24), max(min_side, 24)),
    )

    boxes: list[tuple[int, int, int, int]] = []
    for (fx, fy, fw, fh) in faces:
        # Pad out: the model was trained on crops that include forehead and
        # chin, and a tight box loses the hairline.
        pad_x = int(fw * 0.25)
        pad_y = int(fh * 0.30)
        x1 = max(fx - pad_x, 0)
        y1 = max(fy - pad_y, 0)
        x2 = min(fx + fw + pad_x, width)
        y2 = min(fy + fh + pad_y, height)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    if not boxes:
        return [(0, 0, width, height)]

    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:1] if only_center else boxes


def restore_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Run one aligned 512x512 face through the model."""
    resized = cv2.resize(
        crop_bgr, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    # Convert to RGB float32 in [0, 1].
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # RestoreFormer++ expects [0, 1] range, NCHW — not [-1, 1].
    tensor = rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)

    sess = session()
    output = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]

    # Output is NCHW in [0, 1] range.
    restored = output[0].transpose(1, 2, 0)
    restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)


def blend_back(
    base: np.ndarray,
    restored: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    fidelity: float,
) -> np.ndarray:
    """Paste a restored face back with a feathered edge.

    `fidelity` mixes the result toward the original: 0 is fully restored, 1 is
    untouched. Lower values look more dramatic but drift from the real person,
    which matters when the subject is somebody's grandmother.
    """
    x1, y1, x2, y2 = box
    target_w, target_h = x2 - x1, y2 - y1
    face = cv2.resize(
        restored, (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )

    region = base[y1:y2, x1:x2].astype(np.float32)
    face_f = face.astype(np.float32)
    mixed = face_f * (1.0 - fidelity) + region * fidelity

    # Elliptical feather: a rectangular paste leaves visible seams on the
    # cheeks.
    mask = np.zeros((target_h, target_w), dtype=np.float32)
    cv2.ellipse(
        mask,
        (target_w // 2, target_h // 2),
        (int(target_w * 0.45), int(target_h * 0.48)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    blur = max(3, (min(target_w, target_h) // 12) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]

    base[y1:y2, x1:x2] = (mixed * mask + region * (1.0 - mask)).astype(np.uint8)
    return base


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input") or {}

    try:
        image = decode_image(payload)
        fidelity = float(payload.get("fidelity", 0.5))
        fidelity = min(max(fidelity, 0.0), 1.0)
        only_center = bool(payload.get("only_center_face", False))

        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        boxes = find_faces(bgr, only_center=only_center)
        for box in boxes:
            x1, y1, x2, y2 = box
            restored = restore_crop(bgr[y1:y2, x1:x2])
            bgr = blend_back(bgr, restored, box, fidelity=fidelity)

        result = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return {"image": encode_image(result), "faces": len(boxes)}

    except BadInput as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surface anything else as a job error
        return {"error": f"Face restoration failed: {exc}"}


# Preload at startup so the model is in memory before the first request, and
# FlashBoot can pause the worker with it already resident.
try:
    session()
    detector()
    print("[lumina] RestoreFormer++ preloaded", flush=True)
except Exception as exc:
    print(f"[lumina] preload failed: {exc}", flush=True)

runpod.serverless.start({"handler": handler})
