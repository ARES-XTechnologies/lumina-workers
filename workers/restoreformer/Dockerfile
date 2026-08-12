FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_ROOT=/app/weights

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake RestoreFormer++ ONNX (285MB) into the image
RUN mkdir -p /app/weights/restoreformer && \
    wget -q --header="Authorization: Bearer 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" \
    -O /app/weights/restoreformer/RestoreFormerPlusPlus.onnx \
    "https://huggingface.co/dnnagy/RestoreFormerPlusPlus/resolve/main/RestoreFormerPlusPlus.onnx"

# Bake face detector weights (tiny, ~2MB total)
RUN mkdir -p /app/weights/facedet && \
    wget -q -O /app/weights/facedet/deploy.prototxt \
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt" && \
    wget -q -O /app/weights/facedet/res10_300x300_ssd.caffemodel \
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd.caffemodel"

COPY common.py handler.py ./

CMD ["python3", "-u", "handler.py"]
