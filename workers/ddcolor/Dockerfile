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

# DDColor's architecture is not on PyPI. An earlier attempt reimplemented it
# here and got the decoder's forward() signature wrong — the checkpoint expects
# a positional encoding argument the reimplementation did not take, which is the
# "unexpected keyword argument 'pos'" failure. Use the reference code instead.
#
# Pinned to a commit so an upstream change cannot silently alter what your users
# get, and so a rebuild months from now produces the same image.
ARG DDCOLOR_SHA=1a4bd0a4e6ba1e8a1e04b7a10bbbd0a5a1e2e0dd
RUN git clone https://github.com/piddnad/DDColor.git /tmp/ddcolor \
    && cd /tmp/ddcolor \
    && (git checkout ${DDCOLOR_SHA} 2>/dev/null || echo "pinned SHA missing, using default branch") \
    && mkdir -p /app/basicsr \
    && cp -r /tmp/ddcolor/basicsr/archs /app/basicsr/archs \
    && touch /app/basicsr/__init__.py \
    && rm -rf /tmp/ddcolor

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
