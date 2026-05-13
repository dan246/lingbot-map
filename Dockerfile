FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/hf_cache \
    TORCH_HOME=/app/torch_cache \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential ninja-build \
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

# NVIDIA Kaolin — needed by demo_render rendering pipeline (must match torch 2.8 + cu128)
RUN pip install --index-url https://pypi.org/simple \
        kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

# onnxruntime-gpu for sky segmentation in render pipeline
RUN pip install onnxruntime-gpu

# Project metadata + source needed for editable install
COPY pyproject.toml /app/
COPY lingbot_map /app/lingbot_map
COPY demo.py /app/demo.py
COPY demo_render /app/demo_render

# Project + visualization + rendering extras
RUN pip install -e ".[vis,render]"

# Build the CUDA extensions used by demo_render (voxel_morton_ext, frustum_cull_ext)
RUN cd /app/demo_render/render_cuda_ext && python setup.py build_ext --inplace

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
