FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

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
ARG HF_TOKEN
RUN mkdir -p /app/weights/birefnet && \
    for i in 1 2 3; do \
        wget -q --header="Authorization: Bearer ${HF_TOKEN}" \
        -O /app/weights/birefnet/model.safetensors \
        "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors" \
        && break || sleep 30; \
    done

RUN for f in config.json birefnet.py BiRefNet_config.py; do \
    for i in 1 2 3; do \
        wget -q --header="Authorization: Bearer ${HF_TOKEN}" \
        -O /app/weights/birefnet/$f \
        "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/$f" \
        && break || sleep 10; \
    done; \
    done

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
