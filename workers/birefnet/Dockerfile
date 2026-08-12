FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_ROOT=/app/weights \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake BiRefNet weights + config files (424MB + tiny configs)
RUN mkdir -p /app/weights/birefnet

RUN wget -q --header="Authorization: Bearer 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" \
    -O /app/weights/birefnet/model.safetensors \
    "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors"

RUN for f in config.json birefnet.py BiRefNet_config.py; do \
    wget -q --header="Authorization: Bearer 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" \
    -O /app/weights/birefnet/$f \
    "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/$f"; \
    done

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
