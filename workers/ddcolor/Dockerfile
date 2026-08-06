FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_ROOT=/runpod-volume

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# DDColor's architecture is not on PyPI, so it comes from the reference repo at
# build time. Pinned to a commit so a future upstream change cannot silently
# alter what your users get.
ARG DDCOLOR_REF=main
RUN git clone --depth 1 --branch ${DDCOLOR_REF} \
        https://github.com/piddnad/DDColor.git /tmp/ddcolor \
    && cp -r /tmp/ddcolor/basicsr /app/basicsr 2>/dev/null || true \
    && cp /tmp/ddcolor/basicsr/archs/ddcolor_arch.py /app/ddcolor_arch.py \
    && cp -r /tmp/ddcolor/basicsr/archs/ddcolor_arch_utils /app/ddcolor_arch_utils \
    && rm -rf /tmp/ddcolor

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
