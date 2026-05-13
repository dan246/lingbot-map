# LingBot-Map Docker

A containerised web service that wraps `demo.py` with a browser UI for uploading videos / picking example scenes, and exposes the Viser 3D viewer.

## Host prerequisites

- Docker (>= 25.0) with `nvidia` runtime (NVIDIA Container Toolkit)
- NVIDIA driver supporting CUDA 12.8 (>= 555.x)
- An NVIDIA GPU with enough VRAM (RTX 4090 / A100 / etc.)

No Python or PyTorch needed on the host.

## Layout

```
.
├── Dockerfile              # CUDA 12.8 + Python 3.10 + PyTorch 2.8 + project + webui
├── docker-compose.yml      # GPU passthrough, ports 8000/8080, volume mounts
├── webui/
│   ├── app.py              # FastAPI app
│   ├── templates/          # Jinja2 HTML
│   └── static/             # CSS
├── checkpoints/            # Place .pt files here (mounted read-only)
├── example/                # Example scenes shipped with the repo
├── uploads/                # User-uploaded videos (persisted)
├── logs/                   # Per-job stdout/stderr logs (persisted)
├── hf_cache/, torch_cache/ # HF / Torch download caches (persisted)
└── demo.py, lingbot_map/   # original project
```

## Quick start

```bash
# 1. Make sure a checkpoint is in ./checkpoints/
ls checkpoints/             # should include lingbot-map-long.pt

# 2. Build (first time only, ~10 min: pulls CUDA + torch wheels)
docker compose build

# 3. Start
docker compose up -d

# 4. Open the UI
xdg-open http://localhost:8000
```

When a job's status becomes **viewer_ready** in the UI, click **Open 3D viewer** (or hit <http://localhost:8080> directly).

## Configuration

Edit `docker-compose.yml` or set environment variables before `docker compose up`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEBUI_PORT` | `8000` | Host port for the upload UI |
| `VISER_PORT` | `8080` | Host port for the 3D viewer |
| `DEFAULT_CHECKPOINT` | `lingbot-map-long.pt` | Pre-selected checkpoint in dropdowns |

Example (use different host ports):

```bash
WEBUI_PORT=9000 VISER_PORT=9080 docker compose up -d
```

## How it works

- **One job at a time.** The container holds a single `demo.py` subprocess. Starting a new job stops the previous one (so Viser's port 8080 is always available to the latest result).
- **Upload pipeline.** A user uploads a video → it's saved to `uploads/<job_id>/video.<ext>` → `demo.py --video_path …` is invoked → `demo.py` extracts frames, runs inference, and starts Viser on port 8080.
- **Status detection.** The webui watches the subprocess log for `"Starting viser server"` to flip the job status from `running` → `viewer_ready`.
- **SDPA only.** We pass `--use_sdpa` to `demo.py` to avoid FlashInfer as a hard dependency. To enable FlashInfer, add it to the Dockerfile and drop the flag in `webui/app.py`.

## Common operations

```bash
# Tail container logs
docker compose logs -f

# Tail a specific job's log
tail -f logs/<job_id>.log

# Stop everything
docker compose down

# Rebuild after editing webui/* (Dockerfile COPYs webui/ in)
docker compose build && docker compose up -d

# Drop into the container shell
docker compose exec lingbot-map bash
```

## Troubleshooting

- **"nvidia runtime not found"** — Install the NVIDIA Container Toolkit, then `sudo systemctl restart docker`.
- **"could not select device driver "nvidia""** — Same as above.
- **OOM on the GPU** — Reduce `--fps` (fewer frames extracted from videos) or use a shorter clip; tune `--kv_cache_sliding_window` in `webui/app.py` if needed.
- **Viewer page is blank** — Wait until the job status is `viewer_ready`. Viser only starts after inference completes (can take a couple of minutes per scene).
- **Permission errors writing to mounts** — `chown -R $USER:$USER uploads logs hf_cache torch_cache` (the container writes as the same UID as the host user inside the official CUDA image; if there's a mismatch, set `user:` in `docker-compose.yml`).
