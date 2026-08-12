FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_ROOT=/app/weights

RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake Real-ESRGAN x4plus directly into the image.
# 64MB on a local NVMe vs pulling from a network volume every cold start.
RUN mkdir -p /app/weights/realesrgan && \
    for i in 1 2 3 4 5; do \
        wget -q --tries=3 --timeout=60 \
        -O /app/weights/realesrgan/RealESRGAN_x4plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        && echo "Download OK" && break || (echo "Retry $i..." && sleep 15); \
    done && \
    test -s /app/weights/realesrgan/RealESRGAN_x4plus.pth

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
