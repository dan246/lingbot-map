"""Live RTSP streaming for LingBot-Map.

Pulls frames from an RTSP source, runs GCTStream's Phase-1 (scale frames batched)
once, then enters per-frame Phase-2 streaming inference and pushes the resulting
point clouds + camera frustums into a viser viewer incrementally — until the user
presses "Stop streaming" in the viser GUI or the process gets SIGINT / SIGTERM.

The viewer panel mirrors lingbot_map.vis.point_cloud_viewer.PointCloudViewer's
controls where they make sense in live mode: Reset Up / Reset View Direction /
Camera / Display / Screenshot / Export PLY. Playback-related controls
(timestep slider, video saving, GLB export, etc.) are intentionally omitted.
"""
from __future__ import annotations

import argparse
import io
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _write_ply_binary(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an Nx3 float32 points + Nx3 uint8 colors array as binary PLY."""
    assert points.shape[0] == colors.shape[0]
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = np.empty(n, dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ]))
    body["x"] = points[:, 0].astype(np.float32)
    body["y"] = points[:, 1].astype(np.float32)
    body["z"] = points[:, 2].astype(np.float32)
    body["r"] = colors[:, 0]
    body["g"] = colors[:, 1]
    body["b"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(body.tobytes())


class LiveViewer:
    def __init__(self, port: int,
                 max_points_total: int = 5_000_000,
                 max_points_per_frame: int = 1500,
                 frustum_every: int = 5):
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self.max_points_total = max_points_total
        self.max_points_per_frame = max_points_per_frame

        # State
        self._pts: List[np.ndarray] = []
        self._cls: List[np.ndarray] = []
        self._frustum_handles: Dict[int, object] = {}
        self._extrinsics: Dict[int, np.ndarray] = {}
        self.frame_count = 0
        self.stop_event = threading.Event()
        self._scene_lock = threading.Lock()

        # --- GUI ---
        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if client is None:
                return
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

        # Stream folder
        with self.server.gui.add_folder("Stream"):
            self.btn_stop = self.server.gui.add_button("Stop streaming")
            self.btn_stop.on_click(lambda _e: self.stop_event.set())
            self.label = self.server.gui.add_markdown("Initialising…")

        # Reset View Direction folder
        with self.server.gui.add_folder("Reset View Direction"):
            btn_look_at_center = self.server.gui.add_button("Look At Scene Center")
            btn_overview = self.server.gui.add_button("Overview")
            btn_front = self.server.gui.add_button("Front (+Z)")
            btn_back = self.server.gui.add_button("Back (-Z)")
            btn_top = self.server.gui.add_button("Top (-Y)")
            btn_left = self.server.gui.add_button("Left (-X)")
            btn_right = self.server.gui.add_button("Right (+X)")
            btn_first_cam = self.server.gui.add_button("First Camera")
            btn_latest_cam = self.server.gui.add_button("Latest Camera")

        @btn_look_at_center.on_click
        def _(_) -> None:
            center, _scale = self._compute_scene_center_and_scale()
            for client in self.server.get_clients().values():
                client.camera.look_at = tuple(center)

        @btn_overview.on_click
        def _(_) -> None:
            d = np.array([0.5, -0.6, 0.6])
            self._reset_view_to_direction(d / np.linalg.norm(d))

        @btn_front.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([0.0, 0.0, 1.0]))

        @btn_back.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([0.0, 0.0, -1.0]))

        @btn_top.on_click
        def _(_) -> None:
            self._reset_view_to_direction(
                np.array([0.0, -1.0, 0.0]),
                up=np.array([0.0, 0.0, 1.0]),
            )

        @btn_left.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([-1.0, 0.0, 0.0]))

        @btn_right.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([1.0, 0.0, 0.0]))

        @btn_first_cam.on_click
        def _(_) -> None:
            self._move_to_camera_frame(min(self._extrinsics) if self._extrinsics else None)

        @btn_latest_cam.on_click
        def _(_) -> None:
            self._move_to_camera_frame(max(self._extrinsics) if self._extrinsics else None)

        # Video Display folder
        with self.server.gui.add_folder("Video Display"):
            self.show_video_checkbox = self.server.gui.add_checkbox("Show Current Frame", initial_value=True)
            placeholder = np.zeros((90, 160, 3), dtype=np.uint8)
            self.current_frame_image = self.server.gui.add_image(placeholder, label="Current Frame")

        @self.show_video_checkbox.on_update
        def _(_) -> None:
            self.current_frame_image.visible = self.show_video_checkbox.value

        # Display folder
        with self.server.gui.add_folder("Display"):
            self.conf_pct = self.server.gui.add_slider("Drop low-conf %", 0, 95, 1, 30)
            self.point_size = self.server.gui.add_slider("Point Size", 0.0005, 0.05, 0.0005, 0.002)
            self.cam_size = self.server.gui.add_slider("Camera Size", 0.005, 0.5, 0.005, 0.03)
            self.show_camera_checkbox = self.server.gui.add_checkbox("Show Camera", initial_value=True)
            self.frustum_every_slider = self.server.gui.add_slider(
                "Camera Downsample (every Nth frame)", 1, 50, 1, max(1, int(frustum_every)),
            )
            self.max_points_slider = self.server.gui.add_slider(
                "Max Points Per Frame", 100, 10000, 100, max_points_per_frame,
            )

        self.point_size.on_update(lambda _e: self._refresh_cloud())
        self.cam_size.on_update(lambda _e: self._rerender_frustums())
        self.show_camera_checkbox.on_update(lambda _e: self._toggle_frustums())
        self.max_points_slider.on_update(lambda _e: setattr(self, "max_points_per_frame", int(self.max_points_slider.value)))

        # Screenshot folder
        with self.server.gui.add_folder("Screenshot"):
            self.screenshot_button = self.server.gui.add_button("Take Screenshot")
            self.screenshot_resolution = self.server.gui.add_dropdown(
                "Resolution",
                options=["1920x1080", "2560x1440", "3840x2160", "Current"],
                initial_value="1920x1080",
            )
            self.screenshot_path = self.server.gui.add_text(
                "Save Path", initial_value="/app/uploads/live_screenshot.png",
            )
            self.screenshot_status = self.server.gui.add_text("Status", initial_value="Ready")

        @self.screenshot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self._take_screenshot(event.client)

        # Export PLY folder
        with self.server.gui.add_folder("Export Point Cloud"):
            self.ply_path = self.server.gui.add_text(
                "Output Path", initial_value="/app/uploads/live_pointcloud.ply",
            )
            self.export_button = self.server.gui.add_button("Export PLY")
            self.export_status = self.server.gui.add_text("Status", initial_value="Ready")

        @self.export_button.on_click
        def _(_) -> None:
            self._export_ply()

    # ----- scene helpers --------------------------------------------------

    def _compute_scene_center_and_scale(self) -> Tuple[np.ndarray, float]:
        with self._scene_lock:
            if not self._pts:
                return np.zeros(3, dtype=np.float32), 1.0
            all_pts = np.concatenate(self._pts).astype(np.float32)
        if len(all_pts) > 50_000:
            idx = np.random.choice(len(all_pts), 50_000, replace=False)
            all_pts = all_pts[idx]
        center = all_pts.mean(axis=0)
        scale = float(np.linalg.norm(all_pts.std(axis=0))) or 1.0
        return center, scale

    def _reset_view_to_direction(self, direction: np.ndarray,
                                 up: np.ndarray = np.array([0.0, -1.0, 0.0]),
                                 distance_scale: float = 3.0) -> None:
        center, scale = self._compute_scene_center_and_scale()
        position = center + direction * scale * distance_scale
        for client in self.server.get_clients().values():
            client.camera.up_direction = tuple(up)
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(center)

    def _move_to_camera_frame(self, frame_idx: Optional[int]) -> None:
        if frame_idx is None or frame_idx not in self._extrinsics:
            return
        extr = self._extrinsics[frame_idx]
        R_cw = extr[:3, :3]
        t_cw = extr[:3, 3]
        viewing_dir = R_cw[:, 2]
        position = t_cw - viewing_dir * 0.3
        look_at = t_cw + viewing_dir * 0.5
        up = -R_cw[:, 1]
        for client in self.server.get_clients().values():
            client.camera.up_direction = tuple(up)
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(look_at)

    def _toggle_frustums(self) -> None:
        visible = self.show_camera_checkbox.value
        for h in self._frustum_handles.values():
            try:
                h.visible = visible
            except Exception:
                pass

    def _rerender_frustums(self) -> None:
        scale = float(self.cam_size.value)
        for idx, h in list(self._frustum_handles.items()):
            try:
                h.scale = scale
            except Exception:
                pass

    def _take_screenshot(self, client) -> None:
        if client is None:
            self.screenshot_status.value = "No client connected."
            return
        res = self.screenshot_resolution.value
        try:
            if res == "Current":
                img = client.get_render(height=None, width=None)
            else:
                w, h = (int(x) for x in res.split("x"))
                img = client.get_render(height=h, width=w)
        except Exception as e:
            self.screenshot_status.value = f"Error: {e}"
            return
        try:
            out = self.screenshot_path.value or "/app/uploads/live_screenshot.png"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(out)
            self.screenshot_status.value = f"Saved → {out}"
        except Exception as e:
            self.screenshot_status.value = f"Save error: {e}"

    def _export_ply(self) -> None:
        with self._scene_lock:
            if not self._pts:
                self.export_status.value = "Nothing to export yet."
                return
            all_pts = np.concatenate(self._pts).astype(np.float32)
            all_cls = np.concatenate(self._cls).astype(np.uint8)
        out = self.ply_path.value or "/app/uploads/live_pointcloud.ply"
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            _write_ply_binary(out, all_pts, all_cls)
            self.export_status.value = f"Saved {len(all_pts):,} pts → {out}"
        except Exception as e:
            self.export_status.value = f"Error: {e}"

    # ----- frame intake ---------------------------------------------------

    def push_frame(self, points: np.ndarray, colors: np.ndarray, conf: np.ndarray,
                   extrinsic: Optional[np.ndarray] = None,
                   image_thumb: Optional[np.ndarray] = None) -> None:
        if conf is not None and conf.size > 0:
            thresh = np.percentile(conf, self.conf_pct.value)
            m = conf >= thresh
            points = points[m]; colors = colors[m]
        cap = int(self.max_points_per_frame)
        if len(points) > cap:
            idx = np.random.choice(len(points), cap, replace=False)
            points = points[idx]; colors = colors[idx]
        with self._scene_lock:
            self._pts.append(points)
            self._cls.append(colors)
            self.frame_count += 1
            total = sum(len(p) for p in self._pts)
            while total > self.max_points_total and len(self._pts) > 1:
                total -= len(self._pts[0])
                self._pts.pop(0); self._cls.pop(0)

        if image_thumb is not None and self.current_frame_image is not None and self.show_video_checkbox.value:
            try:
                self.current_frame_image.image = image_thumb
            except Exception:
                pass

        if extrinsic is not None:
            self._extrinsics[self.frame_count] = extrinsic
            stride = int(self.frustum_every_slider.value)
            if stride < 1: stride = 1
            if (self.frame_count % stride) == 0:
                self._add_frustum(self.frame_count, extrinsic, image_thumb)

        self._refresh_cloud()

    def _add_frustum(self, idx: int, extr_cw_3x4: np.ndarray, image_thumb: Optional[np.ndarray]):
        R_cw = extr_cw_3x4[:3, :3]
        t_cw = extr_cw_3x4[:3, 3]
        try:
            so3 = tf.SO3.from_matrix(R_cw).wxyz
            h = self.server.scene.add_camera_frustum(
                f"/cams/frame_{idx:06d}",
                fov=np.deg2rad(60.0),
                aspect=1.4,
                scale=float(self.cam_size.value),
                wxyz=so3,
                position=t_cw,
                image=image_thumb,
            )
            try:
                h.visible = self.show_camera_checkbox.value
            except Exception:
                pass
            self._frustum_handles[idx] = h
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


def _slice_pred(pred: dict, i: int) -> dict:
    out = {}
    for k, v in pred.items():
        if torch.is_tensor(v):
            out[k] = v[:, i:i + 1]
    return out


def extract_pointcloud_pose(pred1: dict, img_chw: torch.Tensor
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = pred1["depth"][0, 0, ..., 0].float().cpu().numpy()
    conf = pred1["depth_conf"][0, 0].float().cpu().numpy()
    H, W = depth.shape

    pose_enc = pred1["pose_enc"][0:1, 0:1, :]
    extr_wc, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
    extr_4x4 = torch.zeros((*extr_wc.shape[:-2], 4, 4),
                           device=extr_wc.device, dtype=extr_wc.dtype)
    extr_4x4[..., :3, :4] = extr_wc
    extr_4x4[..., 3, 3] = 1.0
    extr_cw = closed_form_inverse_se3_general(extr_4x4)[..., :3, :4]
    extr_np = extr_cw[0, 0].float().cpu().numpy()
    intr_np = intr[0, 0].float().cpu().numpy()

    world = unproject_depth_map_to_point_map(
        depth.reshape(1, H, W, 1),
        extr_np.reshape(1, 3, 4),
        intr_np.reshape(1, 3, 3),
    )[0]

    rgb = img_chw.float().cpu().numpy().transpose(1, 2, 0)
    colors_u8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return (world.reshape(-1, 3),
            colors_u8.reshape(-1, 3),
            conf.reshape(-1),
            extr_np,
            colors_u8)


def open_rtsp(url: str, retries: int = 3, delay: float = 1.0) -> cv2.VideoCapture:
    import os
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
        "Check: (1) RTSP host reachable from container; "
        "(2) URL correct (host, port, path, credentials); "
        "(3) firewall outbound to that host:port."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp_url", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--scale_frames", type=int, default=8)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=16)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--camera_num_iterations", type=int, default=4,
                        help="Camera head iterative-refinement steps. 1 = faster.")
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--max_points_total", type=int, default=5_000_000)
    parser.add_argument("--max_points_per_frame", type=int, default=1500)
    parser.add_argument("--frustum_every", type=int, default=5)
    parser.add_argument("--warmup_seconds", type=float, default=2.0)
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
        camera_num_iterations=args.camera_num_iterations,
    ).to(device).eval()

    log.info("Loading checkpoint: %s", args.model_path)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info("Loaded. missing=%d unexpected=%d", len(missing), len(unexpected))

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
