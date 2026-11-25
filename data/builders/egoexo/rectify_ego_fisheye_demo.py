#!/usr/bin/env python3
"""Generate quick EgoExo ego fisheye-to-perspective demo images.

This script uses an equidistant fisheye approximation for Aria ego RGB frames.
It is intended for qualitative inspection before integrating full camera calibration.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class RectifyConfig:
    output_size: int
    fisheye_fov_deg: float
    perspective_fov_deg: float
    yaw_deg: float
    pitch_deg: float


def _parse_int_list(text: str) -> list[int]:
    values = [t.strip() for t in text.split(",") if t.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(v) for v in values]


def _parse_float_list(text: str) -> list[float]:
    values = [t.strip() for t in text.split(",") if t.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(v) for v in values]


def _estimate_fisheye_circle(frame: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    valid = gray > 8
    ys, xs = np.where(valid)
    h, w = gray.shape
    if len(xs) < 100:
        return (w / 2.0, h / 2.0, min(h, w) * 0.49)

    cx = float(xs.mean())
    cy = float(ys.mean())
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    radius = 0.5 * float(min(x_max - x_min, y_max - y_min))
    radius = max(10.0, min(radius, min(h, w) * 0.5))
    return (cx, cy, radius)


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    r_yaw = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    return r_pitch @ r_yaw


def fisheye_to_perspective(
    frame_bgr: np.ndarray,
    cfg: RectifyConfig,
    circle: tuple[float, float, float],
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    cx, cy, radius = circle

    out = cfg.output_size
    yy, xx = np.meshgrid(
        np.arange(out, dtype=np.float32),
        np.arange(out, dtype=np.float32),
        indexing="ij",
    )
    c = (out - 1) / 2.0

    f_p = c / np.tan(np.deg2rad(cfg.perspective_fov_deg) / 2.0)
    x_cam = (xx - c) / f_p
    y_cam = (yy - c) / f_p
    z_cam = np.ones_like(x_cam)

    rays = np.stack([x_cam, y_cam, z_cam], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-8

    rot = _rotation_matrix(cfg.yaw_deg, cfg.pitch_deg)
    rays = rays @ rot.T

    theta = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
    phi = np.arctan2(rays[..., 1], rays[..., 0])

    theta_max = np.deg2rad(cfg.fisheye_fov_deg) / 2.0
    f_fish = radius / max(theta_max, 1e-6)
    r = f_fish * theta

    map_x = (cx + r * np.cos(phi)).astype(np.float32)
    map_y = (cy + r * np.sin(phi)).astype(np.float32)

    valid = (
        (map_x >= 0)
        & (map_x <= (w - 1))
        & (map_y >= 0)
        & (map_y <= (h - 1))
        & (r <= radius)
    )

    rect = cv2.remap(
        frame_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rect[~valid] = 0
    return rect


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}.")
    return frame


def _hstack_with_labels(
    images: Iterable[np.ndarray], labels: Iterable[str]
) -> np.ndarray:
    imgs = [img.copy() for img in images]
    labs = list(labels)
    if len(imgs) != len(labs):
        raise ValueError("images and labels length mismatch")

    h = max(im.shape[0] for im in imgs)
    padded = []
    for im, label in zip(imgs, labs):
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(
                im, (int(im.shape[1] * scale), h), interpolation=cv2.INTER_AREA
            )
        canvas = im.copy()
        cv2.putText(
            canvas,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        padded.append(canvas)
    return cv2.hconcat(padded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create quick ego fisheye rectification demos."
    )
    parser.add_argument(
        "--ego-video",
        type=str,
        default="/data/egoexo/takes/cmu_bike01_2/frame_aligned_videos/aria01_214-1.mp4",
        help="Path to ego fisheye-like Aria RGB video (e.g., aria01_214-1.mp4).",
    )
    parser.add_argument(
        "--exo-video",
        type=str,
        default="/data/egoexo/takes/cmu_bike01_2/frame_aligned_videos/cam01.mp4",
        help="Optional exo video for side-by-side reference.",
    )
    parser.add_argument(
        "--frame-indices",
        type=str,
        default="300,1200",
        help="Comma-separated frame indices to export.",
    )
    parser.add_argument(
        "--yaw-deg",
        type=str,
        default="0,35",
        help="Comma-separated yaw values for perspective views.",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=0.0,
        help="Pitch for perspective view (deg).",
    )
    parser.add_argument(
        "--fisheye-fov-deg",
        type=float,
        default=150.0,
        help="Assumed fisheye diagonal FoV for equidistant model.",
    )
    parser.add_argument(
        "--perspective-fov-deg",
        type=float,
        default=95.0,
        help="Output perspective horizontal FoV.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=768,
        help="Square output resolution.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="assets/egoexo_rectify_demo",
        help="Directory to write outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_indices = _parse_int_list(args.frame_indices)
    yaw_values = _parse_float_list(args.yaw_deg)

    os.makedirs(args.out_dir, exist_ok=True)

    ego_cap = cv2.VideoCapture(args.ego_video)
    if not ego_cap.isOpened():
        raise RuntimeError(f"Cannot open ego video: {args.ego_video}")

    exo_cap = cv2.VideoCapture(args.exo_video)
    has_exo = exo_cap.isOpened()

    for frame_idx in frame_indices:
        ego_frame = _read_frame(ego_cap, frame_idx)
        exo_frame = _read_frame(exo_cap, frame_idx) if has_exo else None
        circle = _estimate_fisheye_circle(ego_frame)

        perspective_views = []
        perspective_labels = []
        for yaw in yaw_values:
            cfg = RectifyConfig(
                output_size=args.output_size,
                fisheye_fov_deg=args.fisheye_fov_deg,
                perspective_fov_deg=args.perspective_fov_deg,
                yaw_deg=yaw,
                pitch_deg=args.pitch_deg,
            )
            perspective_views.append(fisheye_to_perspective(ego_frame, cfg, circle))
            perspective_labels.append(f"ego_rect_yaw{int(round(yaw))}")

        ego_preview = cv2.resize(
            ego_frame,
            (args.output_size, args.output_size),
            interpolation=cv2.INTER_AREA,
        )
        panel_images = [ego_preview, *perspective_views]
        panel_labels = ["ego_raw", *perspective_labels]

        if exo_frame is not None:
            exo_preview = cv2.resize(
                exo_frame,
                (args.output_size, args.output_size),
                interpolation=cv2.INTER_AREA,
            )
            panel_images.append(exo_preview)
            panel_labels.append("exo_reference")

        panel = _hstack_with_labels(panel_images, panel_labels)

        panel_path = os.path.join(args.out_dir, f"frame_{frame_idx:06d}_panel.jpg")
        cv2.imwrite(panel_path, panel)

        for idx, image in enumerate(perspective_views):
            out_path = os.path.join(
                args.out_dir,
                f"frame_{frame_idx:06d}_ego_rect_{idx}.jpg",
            )
            cv2.imwrite(out_path, image)

    ego_cap.release()
    exo_cap.release()

    print(f"Wrote demo outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
