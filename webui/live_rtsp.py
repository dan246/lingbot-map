"""Live RTSP streaming for LingBot-Map.

Pulls frames from an RTSP source, runs GCTStream's Phase-1 (scale frames batched)
once, then enters per-frame Phase-2 streaming inference and pushes the resulting
point clouds + camera frustums into a viser viewer incrementally — until the user
presses "Stop streaming" in the viser GUI or the process gets SIGINT / SIGTERM.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import viser
import viser.transforms as tf
from PIL import Image

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.geometry import (
    closed_form_inverse_se3_general,
    unproject_depth_map_to_point_map,
)
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("live-rtsp")


# ---------------------------------------------------------------------------
# Preprocessing  (mirrors load_and_preprocess_images(mode="crop"))
# ---------------------------------------------------------------------------

def _pil_to_tensor_crop(pil_rgb: Image.Image, image_size: int, patch_size: int) -> torch.Tensor:
    w, h = pil_rgb.size
    new_w = image_size
    new_h = round(h * (new_w / w) / patch_size) * patch_size
    img = pil_rgb.resize((new_w, new_h), Image.Resampling.BICUBIC)
    t = TF.to_tensor(img)
    if new_h > image_size:
        s = (new_h - image_size) // 2
        t = t[:, s:s + image_size, :]
    return t


def cv2_to_tensor(bgr: np.ndarray, image_size: int, patch_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _pil_to_tensor_crop(Image.fromarray(rgb), image_size, patch_size)


# ---------------------------------------------------------------------------
# Live viewer
# ---------------------------------------------------------------------------

class LiveViewer:
    def __init__(self, port: int,
                 max_points_total: int = 5_000_000,
                 max_points_per_frame: int = 1500,
                 frustum_every: int = 5):
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self.max_points_total = max_points_total
        self.max_points_per_frame = max_points_per_frame
        self.frustum_every = max(1, int(frustum_every))

        self._pts: List[np.ndarray] = []
        self._cls: List[np.ndarray] = []
        self.frame_count = 0
        self.stop_event = threading.Event()
        self._scene_lock = threading.Lock()

        with self.server.gui.add_folder("Stream"):
            self.btn_stop = self.server.gui.add_button("Stop streaming")
            self.btn_stop.on_click(lambda _e: self.stop_event.set())
            self.label = self.server.gui.add_markdown("Initialising…")
        with self.server.gui.add_folder("Display"):
            self.conf_pct = self.server.gui.add_slider("Drop low-conf %", 0, 95, 1, 30)
            self.point_size = self.server.gui.add_slider("Point size", 0.0005, 0.01, 0.0005, 0.002)
            self.point_size.on_update(lambda _e: self._refresh_cloud())

    def push_frame(self, points: np.ndarray, colors: np.ndarray, conf: np.ndarray,
                   extrinsic: Optional[np.ndarray] = None,
                   image_thumb: Optional[np.ndarray] = None) -> None:
        if conf is not None and conf.size > 0:
            thresh = np.percentile(conf, self.conf_pct.value)
            m = conf >= thresh
            points = points[m]; colors = colors[m]
        if len(points) > self.max_points_per_frame:
            idx = np.random.choice(len(points), self.max_points_per_frame, replace=False)
            points = points[idx]; colors = colors[idx]
        with self._scene_lock:
            self._pts.append(points)
            self._cls.append(colors)
            self.frame_count += 1
            # Cap memory: drop oldest accumulated frames if over budget.
            total = sum(len(p) for p in self._pts)
            while total > self.max_points_total and len(self._pts) > 1:
                total -= len(self._pts[0])
                self._pts.pop(0); self._cls.pop(0)

        # Add a small camera frustum every N frames so the trajectory is visible.
        if extrinsic is not None and (self.frame_count % self.frustum_every == 0):
            self._add_frustum(self.frame_count, extrinsic, image_thumb)

        self._refresh_cloud()

    def _add_frustum(self, idx: int, extr_cw_3x4: np.ndarray, image_thumb: Optional[np.ndarray]):
        # extr_cw_3x4 is already cam->world (3,4).
        R_cw = extr_cw_3x4[:3, :3]
        t_cw = extr_cw_3x4[:3, 3]
        try:
            so3 = tf.SO3.from_matrix(R_cw).wxyz
            self.server.scene.add_camera_frustum(
                f"/cams/frame_{idx:06d}",
                fov=np.deg2rad(60.0),
                aspect=1.4,
                scale=0.03,
                wxyz=so3,
                position=t_cw,
                image=image_thumb,
            )
        except Exception as e:
            log.debug("frustum add failed: %s", e)

    def _refresh_cloud(self):
        with self._scene_lock:
            if not self._pts:
                return
            all_pts = np.concatenate(self._pts).astype(np.float32)
            all_cls = np.concatenate(self._cls).astype(np.uint8)
        self.server.scene.add_point_cloud(
            "/live",
            points=all_pts,
            colors=all_cls,
            point_size=self.point_size.value,
            point_shape="circle",
        )
        self.label.content = f"Frames: **{self.frame_count}** · Points: **{len(all_pts):,}**"


# ---------------------------------------------------------------------------
# Predictions -> point cloud / pose
# ---------------------------------------------------------------------------

def _slice_pred(pred: dict, i: int) -> dict:
    """Return a per-frame view of a (B=1, S, ...) prediction dict, at index i."""
    out = {}
    for k, v in pred.items():
        if torch.is_tensor(v):
            out[k] = v[:, i:i + 1]
    return out


def extract_pointcloud_pose(pred1: dict, img_chw: torch.Tensor
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For a single-frame prediction, return (points, colors, conf, extr_3x4, image_HWC_uint8)."""
    depth = pred1["depth"][0, 0, ..., 0].float().cpu().numpy()   # (H, W)
    conf = pred1["depth_conf"][0, 0].float().cpu().numpy()        # (H, W)
    H, W = depth.shape

    pose_enc = pred1["pose_enc"][0:1, 0:1, :]                     # [1,1,9]
    extr_wc, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
    # demo.py uses pose_encoding_to_extri_intri's output and then INVERTS it via
    # closed_form_inverse_se3_general to obtain cam->world, which is what
    # unproject_depth_map_to_point_map expects. Replicate that here, otherwise
    # every frame's points are placed at the wrong world location and the
    # accumulated cloud is garbage.
    extr_4x4 = torch.zeros((*extr_wc.shape[:-2], 4, 4),
                           device=extr_wc.device, dtype=extr_wc.dtype)
    extr_4x4[..., :3, :4] = extr_wc
    extr_4x4[..., 3, 3] = 1.0
    extr_cw = closed_form_inverse_se3_general(extr_4x4)[..., :3, :4]
    extr_np = extr_cw[0, 0].float().cpu().numpy()                 # (3,4) cam->world
    intr_np = intr[0, 0].float().cpu().numpy()                    # (3,3)

    world = unproject_depth_map_to_point_map(
        depth.reshape(1, H, W, 1),
        extr_np.reshape(1, 3, 4),
        intr_np.reshape(1, 3, 3),
    )[0]                                                          # (H, W, 3)

    rgb = img_chw.float().cpu().numpy().transpose(1, 2, 0)        # (H, W, 3) in [0,1]
    colors_u8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return (world.reshape(-1, 3),
            colors_u8.reshape(-1, 3),
            conf.reshape(-1),
            extr_np,
            colors_u8)


# ---------------------------------------------------------------------------
# RTSP capture
# ---------------------------------------------------------------------------

def open_rtsp(url: str, retries: int = 3, delay: float = 1.0) -> cv2.VideoCapture:
    """Open an RTSP stream. Forces TCP transport and a short timeout so that
    unreachable hosts fail fast instead of hanging on FFmpeg's 30s default."""
    import os
    # Override capture options for THIS process: force TCP, 5s socket timeout (μs).
    # Format is "key;val|key;val". OpenCV reads this env var per-VideoCapture call.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000|timeout;5000000",
    )
    cap = None
    for attempt in range(1, retries + 1):
        log.info("Opening RTSP (attempt %d/%d): %s", attempt, retries, url)
        t0 = time.time()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            log.info("RTSP opened in %.1fs", time.time() - t0)
            return cap
        log.warning("attempt %d failed after %.1fs", attempt, time.time() - t0)
        if cap is not None:
            cap.release()
        time.sleep(delay)
    raise RuntimeError(
        f"Failed to open RTSP after {retries} attempts: {url}. "
        "Check: (1) RTSP host reachable from container — try "
        "`docker compose exec lingbot-map ffprobe -rtsp_transport tcp -i <url>`; "
        "(2) URL correct (host, port, path, credentials); "
        "(3) firewall outbound to that host:port."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp_url", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--scale_frames", type=int, default=8)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=16,
                        help="Number of recent frames kept in KV cache (each frame ~96MB "
                             "across 24 layers in bf16). 64 = ~6GB just for cache; 16 = ~1.5GB.")
    parser.add_argument("--max_frame_num", type=int, default=1024,
                        help="Upper bound used to preallocate KV cache / RoPE. "
                             "DO NOT bump arbitrarily — memory scales with this.")
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Safety cap on total streamed frames (0 = no cap)")
    parser.add_argument("--max_points_total", type=int, default=5_000_000)
    parser.add_argument("--max_points_per_frame", type=int, default=1500)
    parser.add_argument("--frustum_every", type=int, default=5)
    parser.add_argument("--warmup_seconds", type=float, default=2.0,
                        help="Discard RTSP frames for this many seconds at start so "
                             "the HEVC decoder syncs to a keyframe — skips garbled "
                             "frames that would otherwise corrupt scale estimation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    log.info("Building model…")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.scale_frames,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=4,
    ).to(device).eval()

    log.info("Loading checkpoint: %s", args.model_path)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info("Loaded. missing=%d unexpected=%d", len(missing), len(unexpected))

    # Start viser BEFORE opening RTSP so the webui sees "viewer_ready" early
    # and the user has a tab to watch while frames warm up.
    log.info("Starting viser server on port %d", args.port)
    print(f"Starting viser server on port {args.port}", flush=True)
    viewer = LiveViewer(
        port=args.port,
        max_points_total=args.max_points_total,
        max_points_per_frame=args.max_points_per_frame,
        frustum_every=args.frustum_every,
    )
    log.info("3D viewer at http://localhost:%d", args.port)

    def _on_sig(_signum, _frame):
        log.info("Received signal %s, stopping…", _signum)
        viewer.stop_event.set()
    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    viewer.label.content = "Opening RTSP stream…"
    try:
        cap = open_rtsp(args.rtsp_url)
    except RuntimeError as e:
        log.error(str(e))
        viewer.label.content = f"❌ {e}"
        sys.exit(2)

    # Warmup: discard frames for N seconds so the HEVC decoder picks up an IDR
    # frame. Otherwise the scale frames are half-corrupt and the entire
    # reconstruction gets the wrong scale.
    if args.warmup_seconds > 0:
        log.info("Warming up RTSP decoder for %.1fs…", args.warmup_seconds)
        viewer.label.content = f"Warming up decoder ({args.warmup_seconds:.0f}s)…"
        warm_end = time.time() + args.warmup_seconds
        discarded = 0
        while time.time() < warm_end and not viewer.stop_event.is_set():
            ret, _ = cap.read()
            if ret:
                discarded += 1
            else:
                time.sleep(0.01)
        log.info("Warmup done. Discarded %d frames.", discarded)

    # Phase 1: collect scale_frames frames, then run once.
    viewer.label.content = f"Collecting {args.scale_frames} scale frames…"
    scale_imgs: List[torch.Tensor] = []
    while len(scale_imgs) < args.scale_frames and not viewer.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue
        scale_imgs.append(cv2_to_tensor(frame, args.image_size, args.patch_size))

    if viewer.stop_event.is_set():
        cap.release(); return

    log.info("Phase 1: scale estimation")
    model.clean_kv_cache()
    scale_batch = torch.stack(scale_imgs).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        torch.compiler.cudagraph_mark_step_begin()
        scale_out = model.forward(
            scale_batch,
            num_frame_for_scale=args.scale_frames,
            num_frame_per_block=args.scale_frames,
            causal_inference=True,
        )
    for i in range(args.scale_frames):
        per = _slice_pred(scale_out, i)
        pts, cls, cf, extr, thumb = extract_pointcloud_pose(per, scale_batch[0, i])
        viewer.push_frame(pts, cls, cf, extrinsic=extr, image_thumb=thumb)
    del scale_out

    log.info("Phase 2: live streaming")
    viewer.label.content = f"Live · {viewer.frame_count} frames"

    last_log = time.time()
    while not viewer.stop_event.is_set():
        if args.max_frames and viewer.frame_count >= args.max_frames:
            log.info("Hit --max_frames cap (%d), stopping.", args.max_frames)
            break
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue
        img = cv2_to_tensor(frame, args.image_size, args.patch_size)
        frame_t = img.unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            torch.compiler.cudagraph_mark_step_begin()
            out = model.forward(
                frame_t,
                num_frame_for_scale=args.scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )
        pts, cls, cf, extr, thumb = extract_pointcloud_pose(out, frame_t[0, 0])
        viewer.push_frame(pts, cls, cf, extrinsic=extr, image_thumb=thumb)
        if time.time() - last_log > 5.0:
            log.info("frames=%d", viewer.frame_count)
            last_log = time.time()
        del out

    log.info("Stopped. Total frames: %d", viewer.frame_count)
    cap.release()


if __name__ == "__main__":
    main()
