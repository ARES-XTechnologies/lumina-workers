FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_ROOT=/app/weights

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Bake RestoreFormer++ ONNX (285MB) into the image
ARG HF_TOKEN
RUN mkdir -p /app/weights/restoreformer && \
    for i in 1 2 3; do \
        wget -q --tries=3 --timeout=60 --header="Authorization: Bearer ${HF_TOKEN}" \
        -O /app/weights/restoreformer/RestoreFormerPlusPlus.onnx \
        "https://huggingface.co/dnnagy/RestoreFormerPlusPlus/resolve/main/RestoreFormerPlusPlus.onnx" \
        && break || sleep 30; \
    done && \
    test -s /app/weights/restoreformer/RestoreFormerPlusPlus.onnx


COPY common.py handler.py ./

CMD ["python3", "-u", "handler.py"]
