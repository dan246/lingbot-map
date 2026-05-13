import asyncio
import os
import re
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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

app = FastAPI(title="LingBot-Map Web UI")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

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


def _opt_int(v, fld) -> Optional[int]:
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    try: return int(s)
    except ValueError: raise HTTPException(400, f"{fld} must be int, got {v!r}")


def _opt_float(v, fld) -> Optional[float]:
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    try: return float(s)
    except ValueError: raise HTTPException(400, f"{fld} must be float, got {v!r}")


def _kf_interval(raw: str) -> Optional[int]:
    if raw is None: return None
    s = raw.strip().lower()
    if s in ("", "auto", "0"): return None
    try: v = int(s)
    except ValueError:
        raise HTTPException(400, f"invalid keyframe_interval: {raw!r}")
    if v < 1:
        raise HTTPException(400, "keyframe_interval must be >= 1 (or empty/auto)")
    return v


def _validate_mode(m: str) -> str:
    if m not in ("streaming", "windowed"):
        raise HTTPException(400, f"mode must be 'streaming' or 'windowed', got {m!r}")
    return m


def _build_demo_args(f: dict, source_kind: str, source_value: str) -> List[str]:
    args: List[str] = ["--use_sdpa"]
    if source_kind == "image_folder":
        args += ["--image_folder", source_value]
    elif source_kind == "video_path":
        args += ["--video_path", source_value]
    else:
        raise ValueError(f"unknown source_kind {source_kind}")
    if f.get("fps") is not None: args += ["--fps", str(f["fps"])]
    if f.get("first_k"): args += ["--first_k", str(f["first_k"])]
    if f.get("stride"): args += ["--stride", str(f["stride"])]
    if f.get("rotate_clockwise_90"): args += ["--rotate_clockwise_90"]
    if f.get("image_size"): args += ["--image_size", str(f["image_size"])]
    if f.get("patch_size"): args += ["--patch_size", str(f["patch_size"])]
    if f.get("mode"): args += ["--mode", f["mode"]]
    if f.get("enable_3d_rope"): args += ["--enable_3d_rope"]
    if f.get("max_frame_num"): args += ["--max_frame_num", str(f["max_frame_num"])]
    if f.get("num_scale_frames"): args += ["--num_scale_frames", str(f["num_scale_frames"])]
    if f.get("keyframe_interval") is not None: args += ["--keyframe_interval", str(f["keyframe_interval"])]
    if f.get("kv_cache_sliding_window"): args += ["--kv_cache_sliding_window", str(f["kv_cache_sliding_window"])]
    if f.get("camera_num_iterations"): args += ["--camera_num_iterations", str(f["camera_num_iterations"])]
    if f.get("compile"): args += ["--compile"]
    if f.get("offload_to_cpu") == "yes": args += ["--offload_to_cpu"]
    elif f.get("offload_to_cpu") == "no": args += ["--no-offload_to_cpu"]
    if f.get("window_size"): args += ["--window_size", str(f["window_size"])]
    if f.get("overlap_size"): args += ["--overlap_size", str(f["overlap_size"])]
    if f.get("overlap_keyframes"): args += ["--overlap_keyframes", str(f["overlap_keyframes"])]
    if f.get("conf_threshold") is not None: args += ["--conf_threshold", str(f["conf_threshold"])]
    if f.get("downsample_factor"): args += ["--downsample_factor", str(f["downsample_factor"])]
    if f.get("point_size") is not None: args += ["--point_size", str(f["point_size"])]
    if f.get("mask_sky"): args += ["--mask_sky"]
    if f.get("export_preprocessed"): args += ["--export_preprocessed", str(f["export_preprocessed"])]
    if f.get("extra_args"):
        try: args += shlex.split(f["extra_args"])
        except ValueError as e: raise HTTPException(400, f"extra_args parse error: {e}")
    return args


def _build_rtsp_args(f: dict, rtsp_url: str) -> List[str]:
    args: List[str] = ["--rtsp_url", rtsp_url, "--use_sdpa"]
    if f.get("image_size"): args += ["--image_size", str(f["image_size"])]
    if f.get("patch_size"): args += ["--patch_size", str(f["patch_size"])]
    if f.get("num_scale_frames"): args += ["--scale_frames", str(f["num_scale_frames"])]
    if f.get("kv_cache_sliding_window"): args += ["--kv_cache_sliding_window", str(f["kv_cache_sliding_window"])]
    if f.get("max_frame_num"): args += ["--max_frame_num", str(f["max_frame_num"])]
    if f.get("camera_num_iterations"): args += ["--camera_num_iterations", str(f["camera_num_iterations"])]
    if f.get("max_frames"): args += ["--max_frames", str(f["max_frames"])]
    if f.get("max_points_total"): args += ["--max_points_total", str(f["max_points_total"])]
    if f.get("max_points_per_frame"): args += ["--max_points_per_frame", str(f["max_points_per_frame"])]
    if f.get("frustum_every"): args += ["--frustum_every", str(f["frustum_every"])]
    if f.get("warmup_seconds") is not None: args += ["--warmup_seconds", str(f["warmup_seconds"])]
    if f.get("extra_args"):
        try: args += shlex.split(f["extra_args"])
        except ValueError as e: raise HTTPException(400, f"extra_args parse error: {e}")
    return args


async def _start_job(job_id: str, cmd: list, params: dict):
    global current_proc, current_job_id
    async with _run_lock:
        await _stop_current_locked()
        log_path = LOGS_DIR / f"{job_id}.log"
        jobs[job_id]["cmd"] = " ".join(shlex.quote(c) for c in cmd)
        jobs[job_id]["params"] = params
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


def _gather_demo_form(
    checkpoint, fps, mode, mask_sky, keyframe_interval, image_size, patch_size,
    max_frame_num, num_scale_frames, kv_cache_sliding_window, camera_num_iterations,
    compile_flag, offload_to_cpu, window_size, overlap_size, overlap_keyframes,
    conf_threshold, downsample_factor, point_size, first_k, stride, rotate_clockwise_90,
    enable_3d_rope, export_preprocessed, extra_args,
) -> dict:
    _validate_checkpoint(checkpoint)
    return {
        "checkpoint": checkpoint,
        "fps": fps,
        "mode": _validate_mode(mode),
        "mask_sky": mask_sky,
        "keyframe_interval": _kf_interval(keyframe_interval),
        "image_size": _opt_int(image_size, "image_size"),
        "patch_size": _opt_int(patch_size, "patch_size"),
        "max_frame_num": _opt_int(max_frame_num, "max_frame_num"),
        "num_scale_frames": _opt_int(num_scale_frames, "num_scale_frames"),
        "kv_cache_sliding_window": _opt_int(kv_cache_sliding_window, "kv_cache_sliding_window"),
        "camera_num_iterations": _opt_int(camera_num_iterations, "camera_num_iterations"),
        "compile": bool(compile_flag),
        "offload_to_cpu": offload_to_cpu,
        "window_size": _opt_int(window_size, "window_size"),
        "overlap_size": _opt_int(overlap_size, "overlap_size"),
        "overlap_keyframes": _opt_int(overlap_keyframes, "overlap_keyframes"),
        "conf_threshold": _opt_float(conf_threshold, "conf_threshold"),
        "downsample_factor": _opt_int(downsample_factor, "downsample_factor"),
        "point_size": _opt_float(point_size, "point_size"),
        "first_k": _opt_int(first_k, "first_k"),
        "stride": _opt_int(stride, "stride"),
        "rotate_clockwise_90": bool(rotate_clockwise_90),
        "enable_3d_rope": bool(enable_3d_rope),
        "export_preprocessed": (export_preprocessed or "").strip() or None,
        "extra_args": (extra_args or "").strip(),
    }


@app.post("/run-example")
async def run_example(
    scene: str = Form(...),
    checkpoint: str = Form(...),
    fps: int = Form(10),
    mode: str = Form("streaming"),
    mask_sky: bool = Form(False),
    keyframe_interval: str = Form(""),
    image_size: str = Form(""),
    patch_size: str = Form(""),
    max_frame_num: str = Form(""),
    num_scale_frames: str = Form(""),
    kv_cache_sliding_window: str = Form(""),
    camera_num_iterations: str = Form(""),
    compile_flag: bool = Form(False, alias="compile"),
    offload_to_cpu: str = Form(""),
    window_size: str = Form(""),
    overlap_size: str = Form(""),
    overlap_keyframes: str = Form(""),
    conf_threshold: str = Form(""),
    downsample_factor: str = Form(""),
    point_size: str = Form(""),
    first_k: str = Form(""),
    stride: str = Form(""),
    rotate_clockwise_90: bool = Form(False),
    enable_3d_rope: bool = Form(False),
    export_preprocessed: str = Form(""),
    extra_args: str = Form(""),
):
    if scene not in list_examples():
        raise HTTPException(404, f"unknown example scene: {scene}")
    params = _gather_demo_form(
        checkpoint, fps, mode, mask_sky, keyframe_interval, image_size, patch_size,
        max_frame_num, num_scale_frames, kv_cache_sliding_window, camera_num_iterations,
        compile_flag, offload_to_cpu, window_size, overlap_size, overlap_keyframes,
        conf_threshold, downsample_factor, point_size, first_k, stride, rotate_clockwise_90,
        enable_3d_rope, export_preprocessed, extra_args,
    )
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "id": job_id, "kind": "example",
        "source": scene, "checkpoint": checkpoint,
        "status": "queued", "created_at": time.time(),
    }
    cmd = (
        ["python", "/app/demo.py", "--model_path", f"/app/checkpoints/{checkpoint}",
         "--port", str(VIS_PORT_IN_CONTAINER)]
        + _build_demo_args(params, "image_folder", f"/app/example/{scene}")
    )
    asyncio.create_task(_start_job(job_id, cmd, params))
    return RedirectResponse(f"/job/{job_id}", status_code=303)


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    checkpoint: str = Form(...),
    fps: int = Form(10),
    mode: str = Form("streaming"),
    mask_sky: bool = Form(False),
    keyframe_interval: str = Form(""),
    image_size: str = Form(""),
    patch_size: str = Form(""),
    max_frame_num: str = Form(""),
    num_scale_frames: str = Form(""),
    kv_cache_sliding_window: str = Form(""),
    camera_num_iterations: str = Form(""),
    compile_flag: bool = Form(False, alias="compile"),
    offload_to_cpu: str = Form(""),
    window_size: str = Form(""),
    overlap_size: str = Form(""),
    overlap_keyframes: str = Form(""),
    conf_threshold: str = Form(""),
    downsample_factor: str = Form(""),
    point_size: str = Form(""),
    first_k: str = Form(""),
    stride: str = Form(""),
    rotate_clockwise_90: bool = Form(False),
    enable_3d_rope: bool = Form(False),
    export_preprocessed: str = Form(""),
    extra_args: str = Form(""),
):
    params = _gather_demo_form(
        checkpoint, fps, mode, mask_sky, keyframe_interval, image_size, patch_size,
        max_frame_num, num_scale_frames, kv_cache_sliding_window, camera_num_iterations,
        compile_flag, offload_to_cpu, window_size, overlap_size, overlap_keyframes,
        conf_threshold, downsample_factor, point_size, first_k, stride, rotate_clockwise_90,
        enable_3d_rope, export_preprocessed, extra_args,
    )
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
            if not chunk: break
            f.write(chunk); bytes_written += len(chunk)
    if bytes_written == 0:
        raise HTTPException(400, "empty upload")
    jobs[job_id] = {
        "id": job_id, "kind": "upload",
        "source": file.filename, "size_bytes": bytes_written,
        "checkpoint": checkpoint,
        "status": "queued", "created_at": time.time(),
    }
    cmd = (
        ["python", "/app/demo.py", "--model_path", f"/app/checkpoints/{checkpoint}",
         "--port", str(VIS_PORT_IN_CONTAINER)]
        + _build_demo_args(params, "video_path", str(out_path))
    )
    asyncio.create_task(_start_job(job_id, cmd, params))
    return RedirectResponse(f"/job/{job_id}", status_code=303)


@app.post("/start-rtsp")
async def start_rtsp(
    rtsp_url: str = Form(...),
    checkpoint: str = Form(...),
    image_size: str = Form(""),
    patch_size: str = Form(""),
    num_scale_frames: str = Form(""),
    kv_cache_sliding_window: str = Form(""),
    max_frame_num: str = Form(""),
    camera_num_iterations: str = Form(""),
    max_frames: str = Form(""),
    max_points_total: str = Form(""),
    max_points_per_frame: str = Form(""),
    frustum_every: str = Form(""),
    warmup_seconds: str = Form(""),
    extra_args: str = Form(""),
):
    _validate_checkpoint(checkpoint)
    url = rtsp_url.strip()
    if not url:
        raise HTTPException(400, "rtsp_url is required")
    if not url.lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
        raise HTTPException(400, "url must start with rtsp:// (or rtsps/rtmp/http for testing)")
    params = {
        "rtsp_url": url, "checkpoint": checkpoint,
        "image_size": _opt_int(image_size, "image_size"),
        "patch_size": _opt_int(patch_size, "patch_size"),
        "num_scale_frames": _opt_int(num_scale_frames, "num_scale_frames"),
        "kv_cache_sliding_window": _opt_int(kv_cache_sliding_window, "kv_cache_sliding_window"),
        "max_frame_num": _opt_int(max_frame_num, "max_frame_num"),
        "camera_num_iterations": _opt_int(camera_num_iterations, "camera_num_iterations"),
        "max_frames": _opt_int(max_frames, "max_frames"),
        "max_points_total": _opt_int(max_points_total, "max_points_total"),
        "max_points_per_frame": _opt_int(max_points_per_frame, "max_points_per_frame"),
        "frustum_every": _opt_int(frustum_every, "frustum_every"),
        "warmup_seconds": _opt_float(warmup_seconds, "warmup_seconds"),
        "extra_args": (extra_args or "").strip(),
    }
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "id": job_id, "kind": "rtsp",
        "source": url, "checkpoint": checkpoint,
        "status": "queued", "created_at": time.time(),
    }
    cmd = (
        ["python", "/app/webui/live_rtsp.py",
         "--model_path", f"/app/checkpoints/{checkpoint}",
         "--port", str(VIS_PORT_IN_CONTAINER)]
        + _build_rtsp_args(params, url)
    )
    asyncio.create_task(_start_job(job_id, cmd, params))
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
                if now - last_t < min_interval: continue
                last_t = now
                h, w = frame.shape[:2]
                if width and w > width:
                    nh = int(h * width / w)
                    frame = _cv2.resize(frame, (width, nh))
                ok, jpg = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if not ok: continue
                yield boundary + header + jpg.tobytes() + b"\r\n"
        finally:
            cap.release()
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
