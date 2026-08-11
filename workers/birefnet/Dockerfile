FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_ROOT=/runpod-volume \
    # Keep HF from phoning home: the weights are already on the volume, and a
    # blocked network call would just add seconds to every cold start.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py handler.py ./

CMD ["python", "-u", "handler.py"]
