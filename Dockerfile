FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/hf_cache \
    TORCH_HOME=/app/torch_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        ffmpeg libgl1 libglib2.0-0 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch (separate layer — heaviest, cache-friendly)
RUN pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Project metadata + source needed for editable install
COPY pyproject.toml /app/
COPY lingbot_map /app/lingbot_map
COPY demo.py /app/demo.py

# Project + visualization extras
RUN pip install -e ".[vis]"

# Web UI runtime
RUN pip install \
        "fastapi>=0.111" \
        "uvicorn[standard]>=0.30" \
        "python-multipart>=0.0.9" \
        "jinja2>=3.1" \
        "aiofiles>=23.2"

COPY webui /app/webui

EXPOSE 8000 8080

CMD ["uvicorn", "webui.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
