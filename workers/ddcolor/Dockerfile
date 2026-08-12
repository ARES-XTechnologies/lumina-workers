FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_ROOT=/app/weights

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake DDColor model (870MB) — the big one, worth it to avoid network volume reads
ARG HF_TOKEN
RUN mkdir -p /app/weights/ddcolor && \
    for i in 1 2 3; do \
        wget -q --header="Authorization: Bearer ${HF_TOKEN}" \
        -O /app/weights/ddcolor/pytorch_model.bin \
        "https://huggingface.co/piddnad/ddcolor_paper/resolve/main/pytorch_model.bin" \
        && break || sleep 30; \
    done

# DDColor architecture from the reference repo
RUN git clone --depth 1 https://github.com/piddnad/DDColor.git /tmp/ddcolor && \
    mkdir -p /app/basicsr && \
    cp -r /tmp/ddcolor/basicsr/archs /app/basicsr/archs && \
    touch /app/basicsr/__init__.py && \
    rm -rf /tmp/ddcolor

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
