import asyncio
import os
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import cv2 as _cv2

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE = Path(__file__).parent
ROOT = Path("/app")
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXAMPLE_DIR = ROOT / "example"
UPLOADS_DIR = ROOT / "uploads"
LOGS_DIR = ROOT / "logs"
for d in (UPLOADS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

VIS_PORT_IN_CONTAINER = 8080
HOST_VIS_PORT = int(os.environ.get("HOST_VISER_PORT", "8080"))
DEFAULT_CHECKPOINT = os.environ.get("DEFAULT_CHECKPOINT", "")
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ALLOWED_ARCHIVE_EXT = {".zip"}

app = FastAPI(title="LingBot-Map Web UI")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

# In-memory job registry. Order preserved (insertion order).
jobs: Dict[str, dict] = {}
current_proc: Optional[subprocess.Popen] = None
current_job_id: Optional[str] = None
_run_lock = asyncio.Lock()


def list_checkpoints():
    if not CHECKPOINTS_DIR.exists():
        return []
    return sorted(p.name for p in CHECKPOINTS_DIR.glob("*.pt"))


def list_examples():
    if not EXAMPLE_DIR.exists():
        return []
    return sorted(p.name for p in EXAMPLE_DIR.iterdir() if p.is_dir())


async def _stop_current_locked():
    """Caller must hold _run_lock."""
    global current_proc, current_job_id
    if current_proc and current_proc.poll() is None:
        try:
            current_proc.send_signal(signal.SIGINT)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: current_proc.wait(timeout=10)
            )
        except subprocess.TimeoutExpired:
            current_proc.kill()
            current_proc.wait(timeout=5)
        except Exception:
            pass
        if current_job_id and current_job_id in jobs and jobs[current_job_id]["status"] in ("running", "viewer_ready"):
            jobs[current_job_id]["status"] = "stopped"
    current_proc = None
    current_job_id = None


async def _watch_proc(job_id: str, proc: subprocess.Popen, log_path: Path):
    """Poll subprocess; flip status to viewer_ready when viser line appears."""
    viser_re = re.compile(r"Starting viser server on port|3D viewer at http", re.I)
    while proc.poll() is None:
        await asyncio.sleep(1)
        try:
            if jobs[job_id]["status"] == "running" and log_path.exists():
                txt = log_path.read_text(errors="replace")
                if viser_re.search(txt):
                    jobs[job_id]["status"] = "viewer_ready"
        except Exception:
            pass
    rc = proc.returncode
    jobs[job_id]["return_code"] = rc
    jobs[job_id]["ended_at"] = time.time()
    if jobs[job_id]["status"] not in ("stopped",):
        jobs[job_id]["status"] = "ended" if rc == 0 else "failed"


async def _start_job(
    job_id: str, source_kind: str, source_value: str,
    checkpoint: str, mask_sky: bool, fps: int,
    mode: str = "streaming", keyframe_interval: Optional[int] = None,
):
    global current_proc, current_job_id
    async with _run_lock:
        await _stop_current_locked()
        log_path = LOGS_DIR / f"{job_id}.log"
        ckpt_path = f"/app/checkpoints/{checkpoint}"

        if source_kind == "rtsp":
            cmd = [
                "python", "/app/webui/live_rtsp.py",
                "--model_path", ckpt_path,
                "--port", str(VIS_PORT_IN_CONTAINER),
                "--rtsp_url", source_value,
                "--use_sdpa",
            ]
        else:
            cmd = [
                "python", "/app/demo.py",
                "--model_path", ckpt_path,
                "--port", str(VIS_PORT_IN_CONTAINER),
                "--use_sdpa",
                "--fps", str(fps),
                "--mode", mode,
            ]
            if keyframe_interval is not None and keyframe_interval > 0:
                cmd += ["--keyframe_interval", str(keyframe_interval)]
            if source_kind == "image_folder":
                cmd += ["--image_folder", source_value]
            elif source_kind == "video_path":
                cmd += ["--video_path", source_value]
            else:
                raise ValueError(f"unknown source_kind {source_kind}")
            if mask_sky:
                cmd.append("--mask_sky")

        jobs[job_id]["cmd"] = " ".join(cmd)
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = time.time()

        log_f = open(log_path, "w")
        current_proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd="/app",
        )
        current_job_id = job_id

    asyncio.create_task(_watch_proc(job_id, current_proc, log_path))


def _validate_checkpoint(name: str):
    if name not in list_checkpoints():
        raise HTTPException(404, f"checkpoint not found: {name}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    ckpts = list_checkpoints()
    default_ckpt = DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT in ckpts else (ckpts[0] if ckpts else "")
    return templates.TemplateResponse(request, "index.html", {
        "checkpoints": ckpts,
        "default_checkpoint": default_ckpt,
        "examples": list_examples(),
        "jobs": list(jobs.values())[::-1],
        "current_job_id": current_job_id,
        "host_vis_port": HOST_VIS_PORT,
    })


def _parse_keyframe_interval(raw: str) -> Optional[int]:
    """Empty string / 'auto' / 0 -> let demo.py auto-pick. Otherwise positive int."""
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ("", "auto", "0"):
        return None
    try:
        v = int(s)
    except ValueError:
        raise HTTPException(400, f"invalid keyframe_interval: {raw!r}")
    if v < 1:
        raise HTTPException(400, "keyframe_interval must be >= 1 (or empty/auto)")
    return v


def _validate_mode(mode: str) -> str:
    if mode not in ("streaming", "windowed"):
        raise HTTPException(400, f"mode must be 'streaming' or 'windowed', got {mode!r}")
    return mode


@app.post("/run-example")
async def run_example(
    scene: str = Form(...),
    checkpoint: str = Form(...),
    mask_sky: bool = Form(False),
    fps: int = Form(10),
    mode: str = Form("streaming"),
    keyframe_interval: str = Form(""),
):
    if scene not in list_examples():
        raise HTTPException(404, f"unknown example scene: {scene}")
    _validate_checkpoint(checkpoint)
    mode = _validate_mode(mode)
    kf = _parse_keyframe_interval(keyframe_interval)
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "id": job_id, "kind": "example",
        "source": scene, "checkpoint": checkpoint,
        "mask_sky": mask_sky, "fps": fps,
        "mode": mode, "keyframe_interval": kf,
        "status": "queued", "created_at": time.time(),
    }
    asyncio.create_task(_start_job(
        job_id, "image_folder", f"/app/example/{scene}",
        checkpoint, mask_sky, fps, mode, kf,
    ))
    return RedirectResponse(f"/job/{job_id}", status_code=303)


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    checkpoint: str = Form(...),
    mask_sky: bool = Form(False),
    fps: int = Form(10),
    mode: str = Form("streaming"),
    keyframe_interval: str = Form(""),
):
    _validate_checkpoint(checkpoint)
    mode = _validate_mode(mode)
    kf = _parse_keyframe_interval(keyframe_interval)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED_VIDEO_EXT)}")

    job_id = uuid.uuid4().hex[:8]
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / f"video{suffix}"
    bytes_written = 0
    with open(out_path, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            bytes_written += len(chunk)
    if bytes_written == 0:
        raise HTTPException(400, "empty upload")

    jobs[job_id] = {
        "id": job_id, "kind": "upload",
        "source": file.filename, "size_bytes": bytes_written,
        "checkpoint": checkpoint, "mask_sky": mask_sky, "fps": fps,
        "mode": mode, "keyframe_interval": kf,
        "status": "queued", "created_at": time.time(),
    }
    asyncio.create_task(_start_job(
        job_id, "video_path", str(out_path),
        checkpoint, mask_sky, fps, mode, kf,
    ))
    return RedirectResponse(f"/job/{job_id}", status_code=303)


@app.post("/start-rtsp")
async def start_rtsp(
    rtsp_url: str = Form(...),
    checkpoint: str = Form(...),
):
    _validate_checkpoint(checkpoint)
    url = rtsp_url.strip()
    if not url:
        raise HTTPException(400, "rtsp_url is required")
    if not url.lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
        raise HTTPException(400, "url must start with rtsp:// (or rtsps/rtmp/http for testing)")

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "id": job_id, "kind": "rtsp",
        "source": url, "checkpoint": checkpoint,
        "mask_sky": False, "fps": 0,
        "mode": "streaming", "keyframe_interval": None,
        "status": "queued", "created_at": time.time(),
    }
    asyncio.create_task(_start_job(
        job_id, "rtsp", url,
        checkpoint, False, 0, "streaming", None,
    ))
    return RedirectResponse(f"/job/{job_id}", status_code=303)


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_view(request: Request, job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    return templates.TemplateResponse(request, "job.html", {
        "job": jobs[job_id],
        "host_vis_port": HOST_VIS_PORT,
        "is_current": job_id == current_job_id,
    })


@app.get("/api/job/{job_id}")
async def job_status_api(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    log_path = LOGS_DIR / f"{job_id}.log"
    tail = ""
    if log_path.exists():
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            f.seek(max(0, size - 8000))
            tail = f.read().decode("utf-8", errors="replace")
    return JSONResponse({
        **jobs[job_id],
        "log_tail": tail,
        "is_current": job_id == current_job_id,
    })


@app.post("/stop")
async def stop_current():
    async with _run_lock:
        await _stop_current_locked()
    return RedirectResponse("/", status_code=303)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/preview.mjpg")
def preview_mjpg(rtsp_url: str, fps: int = 10, width: int = 640, quality: int = 70):
    """Live MJPEG preview of an RTSP source for sanity-checking the decode.

    Streams JPEG frames as multipart/x-mixed-replace so a plain <img src> in
    the browser shows it as live video. Opens an independent VideoCapture
    (i.e. consumes a second subscription from the RTSP server alongside the
    inference subprocess). When the browser closes the tab, the generator
    exits and the capture is released.
    """
    if not rtsp_url:
        raise HTTPException(400, "rtsp_url required")

    cap = _cv2.VideoCapture(rtsp_url, _cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(502, f"Failed to open RTSP for preview: {rtsp_url}")

    min_interval = 1.0 / max(1, min(fps, 30))

    def gen():
        try:
            boundary = b"--frame\r\n"
            header = b"Content-Type: image/jpeg\r\n\r\n"
            last_t = 0.0
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05); continue
                now = time.time()
                if now - last_t < min_interval:
                    continue
                last_t = now
                h, w = frame.shape[:2]
                if width and w > width:
                    nh = int(h * width / w)
                    frame = _cv2.resize(frame, (width, nh))
                ok, jpg = _cv2.imencode(".jpg", frame,
                                        [_cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if not ok:
                    continue
                yield boundary + header + jpg.tobytes() + b"\r\n"
        finally:
            cap.release()

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
