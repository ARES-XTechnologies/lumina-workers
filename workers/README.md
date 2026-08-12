# Runpod workers

Four Docker images cover all seven of Lumina's tools:

| Image | Serves | Weights it loads |
| --- | --- | --- |
| `upscale` | Photo Enhancer, Artwork, Portrait Enhance, Anime Enhancer | 4xNomos2_hq_dat2, 4xFaceUpSharpDAT, RealESRGAN_x4plus_anime_6B |
| `restoreformer` | Face Restoration | RestoreFormerPlusPlus.onnx |
| `ddcolor` | Photo Colourisation | DDColor (pytorch_model.bin) |
| `birefnet` | Background Removal | BiRefNet (model.safetensors) |

One image serves four tools because [`spandrel`](https://github.com/chaiNNer-org/spandrel)
recognises DAT, DAT2 and RRDBNet from the weight files themselves. The gateway
passes `model: "nomos" | "faceup" | "anime"` and the worker loads the right one.

**Weights are never baked into the images.** They live on the `lumina-models`
network volume, mounted at `/runpod-volume`. A 2GB image is slow to pull on
every cold start, and swapping a model should not mean rebuilding and
repushing.

---

## 1. Finish filling the volume

The models are already there, but two of them need their config files as well —
the weights alone are not loadable.

Start a cheap CPU pod with `lumina-models` attached (same as when you downloaded
the models), open the web terminal, and run:

```bash
export HF=hf_YOUR_TOKEN_HERE

# --- BiRefNet: architecture + config live beside the weights ---------------
cd /workspace/birefnet
for f in config.json birefnet.py BiRefNet_config.py; do
  wget --header="Authorization: Bearer $HF" -O "$f" \
    "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/$f"
done

# --- Face detector for RestoreFormer++ ------------------------------------
# Without this the worker treats the whole frame as one face, which is only
# correct for already-cropped portraits.
mkdir -p /workspace/facedet && cd /workspace/facedet
wget -O deploy.prototxt \
  "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
wget -O res10_300x300_ssd.caffemodel \
  "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

ls -lh /workspace/*/
```

Then **stop the pod**.

Final layout:

```
/workspace (mounted at /runpod-volume on serverless)
├── upscale/       4xNomos2_hq_dat2.safetensors
├── faceupscale/   4xFaceUpSharpDAT.safetensors
├── realesrgan/    RealESRGAN_x4plus_anime_6B.pth
├── restoreformer/ RestoreFormerPlusPlus.onnx
├── ddcolor/       pytorch_model.bin
├── birefnet/      model.safetensors, config.json, birefnet.py, BiRefNet_config.py
└── facedet/       deploy.prototxt, res10_300x300_ssd.caffemodel
```

---

## 2. Build and push

You need Docker installed locally, or use GitHub Actions (section 5).

```sh
# Log in once. ghcr.io is free for public images and has no pull rate limits,
# unlike Docker Hub's 100-per-6-hours on the free plan.
echo YOUR_GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

cd workers

for w in upscale restoreformer ddcolor birefnet; do
  docker build -t ghcr.io/YOUR_GITHUB_USERNAME/lumina-$w:latest ./$w
  docker push ghcr.io/YOUR_GITHUB_USERNAME/lumina-$w:latest
done
```

The `upscale`, `ddcolor` and `birefnet` images are ~6GB (CUDA + torch);
`restoreformer` is ~1GB because ONNX Runtime does not drag torch in. First
build takes a while, later ones reuse cached layers.

---

## 3. Deploy each endpoint

Runpod → **Serverless** → **New Endpoint** → **Deploy from a Docker image**.

For each of the four:

| Setting | Value |
| --- | --- |
| Container image | `ghcr.io/YOUR_USERNAME/lumina-<worker>:latest` |
| GPU | **RTX 4090** (24GB) |
| Active workers | **0** — pay nothing when idle |
| Max workers | **3** to start |
| Idle timeout | **10s** |
| FlashBoot | **On** — cuts cold starts substantially |
| Network volume | **lumina-models** |
| Container disk | 15GB |

The endpoint and the volume must be in the **same datacenter** (US-CA-2), or
the volume will not attach.

`restoreformer` runs happily on a cheaper GPU — it is a 285MB ONNX model. Try
an RTX A4000 or 4000 Ada if 4090s are scarce.

---

## 4. Wire the ids into the gateway

Each endpoint gives you an id. Put them in `backend/.env`:

```ini
RUNPOD_ENDPOINT_UPSCALE=<upscale endpoint id>
RUNPOD_ENDPOINT_FACE_UPSCALE=<upscale endpoint id>     # same image, same id
RUNPOD_ENDPOINT_UPSCALE_ANIME=<upscale endpoint id>    # same image, same id
RUNPOD_ENDPOINT_FACE=<restoreformer endpoint id>
RUNPOD_ENDPOINT_COLOURISE=<ddcolor endpoint id>
RUNPOD_ENDPOINT_MATTE=<birefnet endpoint id>
```

The three upscale variables can all point at the **same** endpoint id — the
worker picks the weights from the `model` parameter the gateway sends. Give
Portrait or Anime their own endpoint later if one tool gets busy enough to
deserve dedicated capacity.

---

## 5. Building without Docker locally

Push `workers/` to a GitHub repo and let Actions build them. Create
`.github/workflows/workers.yml`:

```yaml
name: Build workers
on:
  push:
    paths: ["workers/**"]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    strategy:
      matrix:
        worker: [upscale, restoreformer, ddcolor, birefnet]
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: workers/${{ matrix.worker }}
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/lumina-${{ matrix.worker }}:latest
```

Push once and all four build in parallel on GitHub's runners, free.

---

## 6. Testing an endpoint

Runpod's dashboard has a **Requests** tab where you can paste a test payload:

```json
{
  "input": {
    "image": "<base64 of a small jpeg>",
    "model": "nomos",
    "scale": 2
  }
}
```

A successful run returns `{"output": {"image": "<base64 png>"}}`.

**Expect the first request to take 60–120 seconds.** That is the cold start:
pulling the image, mounting the volume and loading the checkpoint. Later
requests on a warm worker are a few seconds. The Flutter app already handles
this — its poll deadline is six minutes.

---

## 7. Notes worth knowing

**Scale factors.** All three upscalers are natively 4x. The worker composes
other factors from that: 2x is one pass then a Lanczos downscale (sharper than
a plain 2x resize), 8x is two passes then down, 16x is two passes. So 16x is
genuinely two model passes and costs roughly twice the GPU time — which is why
the gateway gates it behind Max.

**Tiling.** A 4x DAT pass on a 12MP photo will not fit in 24GB in one piece, so
the worker runs overlapping 512px tiles and feathers the seams. Memory stays
flat regardless of input size. Lower `tile` if you ever hit OOM on a smaller
GPU.

**Colourisation keeps its detail.** DDColor runs at 512x512, but the worker
keeps only the two chroma channels and takes luminance from the original at
full resolution. A 4000px photo gains colour without becoming a soft 512px
upscale.

**Face restoration is three steps.** Detect faces, restore each aligned 512px
crop, warp back and blend. Running the model on the whole photo instead — the
obvious shortcut — means it sees a face 80 pixels tall and returns mush.
`fidelity` mixes back toward the original: lower is more dramatic but drifts
from the real person, which matters when the subject is somebody's grandmother.

**PNG between hops.** Workers always return PNG, even though the gateway
delivers JPEG. Portrait Enhance chains two models, and JPEG artefacts would
compound at each step.
