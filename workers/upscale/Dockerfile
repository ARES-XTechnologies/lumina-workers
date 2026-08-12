FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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
    wget -q -O /app/weights/realesrgan/RealESRGAN_x4plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
