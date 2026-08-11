# CUDA 12.4 + torch 2.5, matching the driver on Runpod's 4090 fleet.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Weights live on the network volume, never in the image: a 2GB image is
    # slow to pull on every cold start, and swapping a model should not mean
    # rebuilding and repushing.
    MODELS_ROOT=/runpod-volume

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
