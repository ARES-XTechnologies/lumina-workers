FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_ROOT=/runpod-volume

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The DDColor architecture is bundled directly (ddcolor_arch.py) so the image
# does not need a git clone at build time — reliable in CI and much faster.
COPY common.py handler.py ddcolor_arch.py ./

CMD ["python", "-u", "handler.py"]
