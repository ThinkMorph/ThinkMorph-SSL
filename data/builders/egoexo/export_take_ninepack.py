#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import numpy.typing as npt

try:
    projectaria_calibration: Any = import_module("projectaria_tools.core.calibration")
    projectaria_data_provider: Any = import_module(
        "projectaria_tools.core.data_provider"
    )
except ImportError:
    projectaria_calibration = None
    projectaria_data_provider = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rectify_ego_fisheye_demo import (
    RectifyConfig,
    _estimate_fisheye_circle,
    fisheye_to_perspective,
)


@dataclass(frozen=True)
class SelectedFrame:
    frame_idx: int
    laplacian_score: float


ARIA_RGB_LABEL = "camera-rgb"
ImageArray = npt.NDArray[Any]
FeatureArray = npt.NDArray[np.float32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 9-image package per take (3 ego, 3 exo, 3 exo+marker)."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/egoexo"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--take-names",
        type=str,
        default="",
        help="Comma-separated take names. Example: cmu_bike01_2,sfu_basketball012_10",
    )
    parser.add_argument(
        "--take-list-csv",
        type=Path,
        default=None,
        help="CSV with a column 'take_name'.",
    )
    parser.add_argument(
        "--all-takes",
        action="store_true",
        help="Process all takes found in takes.json.",
    )
    parser.add_argument(
        "--start-after",
        type=str,
        default="",
        help="Resume after this take name in the selected take order.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Favor faster candidate search, allowing slightly different frame picks.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the selected take order across this many shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to process from the selected take order.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sample-step", type=int, default=6)
    parser.add_argument("--min-gap", type=int, default=120)
    parser.add_argument("--ego-size", type=int, default=1024)
    parser.add_argument("--score-size", type=int, default=512)
    parser.add_argument("--crop-ratio", type=float, default=0.10)
    parser.add_argument("--marker-radius", type=int, default=42)
    parser.add_argument("--marker-thickness", type=int, default=6)
    parser.add_argument("--marker-cross-size", type=int, default=72)
    parser.add_argument(
        "--ego-rectify-mode",
        type=str,
        choices=("official", "approximate"),
        default="approximate",
        help="Use official Project Aria calibration or legacy approximation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed takes. Default: skip existing takes.",
    )
    parser.add_argument(
        "--allow-no-projection",
        action="store_true",
        help="Allow export even when strict projection dependencies are missing.",
    )
    return parser.parse_args()


def quat_to_rotation(
    qx: float, qy: float, qz: float, qw: float
) -> npt.NDArray[np.float64]:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def kb3_project(
    pc: npt.NDArray[np.float64], intr: dict[str, float]
) -> tuple[float, float] | None:
    x, y, z = float(pc[0]), float(pc[1]), float(pc[2])
    if z <= 1e-9:
        return None

    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    k0, k1, k2, k3 = intr["k0"], intr["k1"], intr["k2"], intr["k3"]

    a = x / z
    b = y / z
    r = math.hypot(a, b)
    theta = math.atan(r)
    th2 = theta * theta
    th3 = theta * th2
    th5 = th3 * th2
    th7 = th5 * th2
    th9 = th7 * th2
    theta_d = theta + k0 * th3 + k1 * th5 + k2 * th7 + k3 * th9

    if r < 1e-12:
        xd, yd = 0.0, 0.0
    else:
        s = theta_d / r
        xd, yd = s * a, s * b

    return fx * xd + cx, fy * yd + cy


def center_crop(image: ImageArray, ratio: float) -> ImageArray:
    if ratio <= 0:
        return image
    h, w = image.shape[:2]
    dy = int(h * ratio)
    dx = int(w * ratio)
    return image[dy : h - dy, dx : w - dx]


def coerce_seconds(value: object, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float, str)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def get_ego_vrs_path(
    take_dir: Path,
    device_name: str = "aria01",
    allow_noimagestreams: bool = True,
) -> Path:
    candidates = [take_dir / f"{device_name}.vrs"]
    if allow_noimagestreams:
        candidates.append(take_dir / f"{device_name}_noimagestreams.vrs")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing Aria VRS calibration source for {device_name} in {take_dir}"
    )


def get_video_frame_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Failed to read video size for {video_path}")
    return width, height


def read_video_frame(video_path: Path, frame_idx: int) -> ImageArray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def make_approximate_ego_rectifier(
    cfg: RectifyConfig,
) -> Callable[[ImageArray], ImageArray]:
    def rectify(frame: ImageArray) -> ImageArray:
        return fisheye_to_perspective(frame, cfg, _estimate_fisheye_circle(frame))

    return rectify


def make_official_ego_rectifier(
    vrs_path: Path,
    source_size: tuple[int, int],
    cfg: RectifyConfig,
) -> Callable[[ImageArray], ImageArray]:
    if projectaria_data_provider is None or projectaria_calibration is None:
        raise RuntimeError(
            "Official ego rectification requires projectaria-tools to be installed."
        )
    if abs(cfg.yaw_deg) > 1e-6 or abs(cfg.pitch_deg) > 1e-6:
        raise ValueError(
            "Official ego rectification currently supports yaw=0 and pitch=0 only."
        )
    if vrs_path.name.endswith("_noimagestreams.vrs"):
        raise RuntimeError(
            "Official ego rectification requires the image-stream VRS (e.g. aria01.vrs), not aria01_noimagestreams.vrs."
        )

    provider = projectaria_data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Failed to open VRS calibration source: {vrs_path}")
    device_calib = provider.get_device_calibration()
    if device_calib is None:
        raise RuntimeError(f"Missing device calibration in VRS: {vrs_path}")

    source_calib = device_calib.get_camera_calib(ARIA_RGB_LABEL)
    if source_calib is None:
        raise RuntimeError(f"Missing {ARIA_RGB_LABEL} calibration in VRS: {vrs_path}")

    scaled_source_calib = source_calib.rescale(source_size, 1)
    half = (cfg.output_size - 1) / 2.0
    focal = half / math.tan(math.radians(cfg.perspective_fov_deg) / 2.0)
    target_calib = projectaria_calibration.get_linear_camera_calibration(
        cfg.output_size,
        cfg.output_size,
        focal,
        ARIA_RGB_LABEL,
        scaled_source_calib.get_transform_device_camera(),
    )
    distort_by_calibration = projectaria_calibration.distort_by_calibration

    def rectify(frame: ImageArray) -> ImageArray:
        rotated = np.rot90(frame, k=3)
        return distort_by_calibration(rotated, target_calib, scaled_source_calib)

    return rectify


def pick_frames(
    ego_video: Path,
    start_idx: int,
    end_idx: int,
    top_k: int,
    sample_step: int,
    min_gap: int,
    score_cfg: RectifyConfig,
    ego_rectifier: Callable[[ImageArray], ImageArray] | None = None,
    fast_mode: bool = False,
) -> list[SelectedFrame]:
    if end_idx < start_idx:
        return []

    rectifier = ego_rectifier or make_approximate_ego_rectifier(score_cfg)

    def collect_candidates(step: int) -> list[tuple[int, float, FeatureArray]]:
        cap = cv2.VideoCapture(str(ego_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open ego video: {ego_video}")

        cands: list[tuple[int, float, FeatureArray]] = []
        frame_idx = 0
        if start_idx > 0 and hasattr(cap, "set"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            frame_idx = start_idx

        use_grab_skip = fast_mode and hasattr(cap, "grab") and hasattr(cap, "retrieve")
        try:
            while True:
                if frame_idx > end_idx:
                    break

                should_sample = frame_idx >= start_idx and (
                    (frame_idx - start_idx) % step == 0
                )

                if use_grab_skip:
                    ok = cap.grab()
                    if not ok:
                        break
                    if not should_sample:
                        frame_idx += 1
                        continue
                    ok, frame = cap.retrieve()
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx < start_idx:
                        frame_idx += 1
                        continue
                    if not should_sample:
                        frame_idx += 1
                        continue

                if not ok or frame is None:
                    frame_idx += 1
                    continue

                rect = rectifier(frame)
                gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                center = gray[h // 8 : (7 * h) // 8, w // 8 : (7 * w) // 8]
                lap = float(cv2.Laplacian(center, cv2.CV_32F).var())

                small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(
                    np.float32
                )
                feat = small.reshape(-1)
                feat = (feat - feat.mean()) / (feat.std() + 1e-6)
                feat = feat / (np.linalg.norm(feat) + 1e-6)
                cands.append((frame_idx, lap, feat))
                frame_idx += 1
        finally:
            cap.release()
        return cands

    if fast_mode:
        candidate_steps = [max(1, sample_step), max(1, sample_step // 2)]
    else:
        candidate_steps = [max(1, sample_step), max(1, sample_step // 2), 1]
    candidate_steps = list(dict.fromkeys(candidate_steps))

    candidates: list[tuple[int, float, FeatureArray]] = []
    for step in candidate_steps:
        cands = collect_candidates(step)
        if len(cands) >= top_k:
            candidates = cands
            break
        if len(cands) > len(candidates):
            candidates = cands

    target_k = min(top_k, len(candidates))
    if target_k == 0:
        return []

    lap_values = np.array([c[1] for c in candidates], dtype=np.float32)
    lap_norm = (lap_values - lap_values.min()) / (np.ptp(lap_values) + 1e-6)

    selected: list[int] = [int(np.argmax(lap_norm))]
    while len(selected) < target_k:
        best_i = None
        best_score = -1e9
        for i, c in enumerate(candidates):
            if i in selected:
                continue
            fidx = c[0]
            if any(abs(fidx - candidates[j][0]) < min_gap for j in selected):
                continue
            novelty = min(1.0 - float(np.dot(c[2], candidates[j][2])) for j in selected)
            score = 0.45 * float(lap_norm[i]) + 0.55 * novelty
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            for i, c in enumerate(candidates):
                if i in selected:
                    continue
                novelty = min(
                    1.0 - float(np.dot(c[2], candidates[j][2])) for j in selected
                )
                score = 0.45 * float(lap_norm[i]) + 0.55 * novelty
                if score > best_score:
                    best_score = score
                    best_i = i

        if best_i is None:
            break
        selected.append(best_i)

    picked = [SelectedFrame(candidates[i][0], candidates[i][1]) for i in selected]
    picked.sort(key=lambda x: x.frame_idx)
    return picked


def load_take_names(args: argparse.Namespace) -> list[str]:
    if args.all_takes:
        return []

    names: list[str] = []
    if args.take_names.strip():
        names.extend([x.strip() for x in args.take_names.split(",") if x.strip()])
    if args.take_list_csv is not None:
        with open(args.take_list_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("take_name", "").strip()
                if name:
                    names.append(name)
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("No takes specified. Use --take-names or --take-list-csv")
    return names


def resolve_take_names(
    args: argparse.Namespace,
    take_by_name: dict[str, dict[str, object]],
) -> list[str]:
    if args.all_takes:
        take_names = sorted(take_by_name.keys())
    else:
        take_names = load_take_names(args)

    start_after = args.start_after.strip()
    if start_after:
        if start_after not in take_names:
            raise ValueError(
                f"--start-after take not found in selected take order: {start_after}"
            )
        start_idx = take_names.index(start_after) + 1
        take_names = take_names[start_idx:]

    num_shards = int(getattr(args, "num_shards", 1))
    shard_index = int(getattr(args, "shard_index", 0))
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if num_shards > 1:
        take_names = [
            take_name
            for idx, take_name in enumerate(take_names)
            if idx % num_shards == shard_index
        ]

    return take_names


def load_global_time_map(
    capture_timesync: Path,
    cache: dict[Path, dict[int, float]] | None = None,
) -> dict[int, float]:
    if cache is not None and capture_timesync in cache:
        return cache[capture_timesync]

    global_map: dict[int, float] = {}
    with open(capture_timesync) as f:
        reader = csv.DictReader(f)
        for row in reader:
            afn = row.get("aria01_214-1_frame_number", "")
            gt = row.get("aria01_214-1_global_time", "")
            if afn and gt:
                global_map[int(float(afn))] = float(gt)

    if cache is not None:
        cache[capture_timesync] = global_map
    return global_map


def is_take_already_processed(take_out: Path) -> bool:
    metadata_json = take_out / "metadata.json"
    metadata_csv = take_out / "metadata.csv"
    if not metadata_json.exists() or not metadata_csv.exists():
        return False
    ego = list(take_out.glob("ego_*.jpg"))
    exo = list(take_out.glob("exo_*.jpg"))
    marked = list(take_out.glob("exo_marked_*.jpg"))
    return len(ego) > 0 and len(exo) > 0 and len(marked) > 0


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_root / "takes.json") as f:
        takes = json.load(f)
    take_by_name = {t["take_name"]: t for t in takes if "take_name" in t}
    take_names = resolve_take_names(args, take_by_name)

    summary: list[dict[str, object]] = []
    timesync_cache: dict[Path, dict[int, float]] = {}

    for take_name in take_names:
        if take_name not in take_by_name:
            print(f"[skip] take not found: {take_name}")
            continue

        take = take_by_name[take_name]
        take_out = out_dir / take_name
        take_out.mkdir(parents=True, exist_ok=True)

        if not args.overwrite and is_take_already_processed(take_out):
            print(f"[skip] already processed: {take_name}")
            continue

        video_dir = args.dataset_root / "takes" / take_name / "frame_aligned_videos"
        ego_video = video_dir / "aria01_214-1.mp4"
        exo_cam = take.get("best_exo", "cam01")
        exo_video = video_dir / f"{exo_cam}.mp4"
        if not exo_video.exists():
            exo_cam = "cam01"
            exo_video = video_dir / "cam01.mp4"

        if not ego_video.exists() or not exo_video.exists():
            print(f"[skip] missing ego/exo video for {take_name}")
            continue

        fps = 30.0
        start_sec = coerce_seconds(take.get("task_start_sec"), 0.0)
        end_sec = coerce_seconds(take.get("task_end_sec"), 0.0)
        start_idx = max(0, int(start_sec * fps))
        end_idx = int(end_sec * fps)

        score_cfg = RectifyConfig(
            output_size=args.score_size,
            fisheye_fov_deg=150.0,
            perspective_fov_deg=95.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
        )
        export_cfg = RectifyConfig(
            output_size=args.ego_size,
            fisheye_fov_deg=150.0,
            perspective_fov_deg=95.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
        )

        ego_source_size = get_video_frame_size(ego_video)
        if args.ego_rectify_mode == "official":
            try:
                ego_vrs = get_ego_vrs_path(
                    args.dataset_root / "takes" / take_name,
                    device_name=ego_video.stem.split("_")[0],
                    allow_noimagestreams=False,
                )
            except FileNotFoundError as exc:
                print(
                    f"[skip] {take_name} missing official ego calibration source: {exc}"
                )
                continue
            score_rectifier = make_official_ego_rectifier(
                ego_vrs,
                source_size=ego_source_size,
                cfg=score_cfg,
            )
            export_rectifier = make_official_ego_rectifier(
                ego_vrs,
                source_size=ego_source_size,
                cfg=export_cfg,
            )
        else:
            score_rectifier = make_approximate_ego_rectifier(score_cfg)
            export_rectifier = make_approximate_ego_rectifier(export_cfg)

        selected = pick_frames(
            ego_video=ego_video,
            start_idx=start_idx,
            end_idx=end_idx,
            top_k=args.top_k,
            sample_step=args.sample_step,
            min_gap=args.min_gap,
            score_cfg=score_cfg,
            ego_rectifier=score_rectifier,
            fast_mode=args.fast_mode,
        )
        if len(selected) == 0:
            print(f"[skip] insufficient candidates for {take_name} (need >=1, got 0)")
            continue
        if len(selected) < args.top_k:
            print(
                f"[warn] {take_name} has only {len(selected)} valid frames "
                f"(< requested {args.top_k}); exporting fewer images"
            )

        capture_name = take["capture"]["capture_name"]
        projection_ready = True
        projection_reason = "ok"

        global_map: dict[int, float] = {}
        timesync_start_idx = int(take["timesync_start_idx"])
        timesync_end_idx = int(take["timesync_end_idx"])
        g0 = None
        g1 = None
        traj_ts_arr = np.array([], dtype=np.float64)
        traj_xyz_arr = np.array([], dtype=np.float64)
        ts0 = 0.0
        ts1 = 0.0
        intr: dict[str, float] = {}
        t_world_cam = np.zeros(3, dtype=np.float64)
        r_cam_world = np.eye(3, dtype=np.float64)

        capture_timesync = (
            args.dataset_root / "captures" / capture_name / "timesync.csv"
        )
        traj_csv = (
            args.dataset_root
            / "takes"
            / take_name
            / "trajectory"
            / "closed_loop_trajectory.csv"
        )
        calib_csv = (
            args.dataset_root / "takes" / take_name / "trajectory" / "gopro_calibs.csv"
        )

        if (
            not capture_timesync.exists()
            or not traj_csv.exists()
            or not calib_csv.exists()
        ):
            projection_ready = False
            projection_reason = "missing_timesync_or_trajectory_or_calibration"
        else:
            global_map = load_global_time_map(capture_timesync, timesync_cache)
            g0 = global_map.get(timesync_start_idx)
            g1 = global_map.get(timesync_end_idx)
            if g0 is None or g1 is None or g1 <= g0:
                projection_ready = False
                projection_reason = "invalid_take_global_time_range"

        if projection_ready:
            traj_ts: list[float] = []
            traj_xyz: list[tuple[float, float, float]] = []
            with open(traj_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    traj_ts.append(float(row["tracking_timestamp_us"]))
                    traj_xyz.append(
                        (
                            float(row["tx_world_device"]),
                            float(row["ty_world_device"]),
                            float(row["tz_world_device"]),
                        )
                    )
            if len(traj_ts) == 0:
                projection_ready = False
                projection_reason = "empty_trajectory"
            else:
                traj_ts_arr = np.array(traj_ts, dtype=np.float64)
                traj_xyz_arr = np.array(traj_xyz, dtype=np.float64)
                ts0 = float(traj_ts_arr[0])
                ts1 = float(traj_ts_arr[-1])

        if projection_ready:
            calib = None
            with open(calib_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["cam_uid"] == exo_cam:
                        calib = row
                        break
            if calib is None:
                projection_ready = False
                projection_reason = "missing_exo_calibration_row"
            else:
                intr = {
                    "fx": float(calib["intrinsics_0"]),
                    "fy": float(calib["intrinsics_1"]),
                    "cx": float(calib["intrinsics_2"]),
                    "cy": float(calib["intrinsics_3"]),
                    "k0": float(calib["intrinsics_4"]),
                    "k1": float(calib["intrinsics_5"]),
                    "k2": float(calib["intrinsics_6"]),
                    "k3": float(calib["intrinsics_7"]),
                }

                t_world_cam = np.array(
                    [
                        float(calib["tx_world_cam"]),
                        float(calib["ty_world_cam"]),
                        float(calib["tz_world_cam"]),
                    ],
                    dtype=np.float64,
                )
                qx = float(calib["qx_world_cam"])
                qy = float(calib["qy_world_cam"])
                qz = float(calib["qz_world_cam"])
                qw = float(calib["qw_world_cam"])
                r_world_cam = quat_to_rotation(qx, qy, qz, qw)
                r_cam_world = r_world_cam.T

        if not projection_ready:
            print(f"[warn] {take_name} projection disabled: {projection_reason}")
            if not args.allow_no_projection:
                print(f"[skip] {take_name} requires strict projection; skipped")
                continue

        frame_meta: list[dict[str, object]] = []
        for rank, sf in enumerate(selected, start=1):
            frame_idx = sf.frame_idx

            ego_frame = read_video_frame(ego_video, frame_idx)
            exo_frame = read_video_frame(exo_video, frame_idx)

            ego_rect = export_rectifier(ego_frame)
            ego_crop = center_crop(ego_rect, args.crop_ratio)

            capture_aria_idx = timesync_start_idx + frame_idx
            global_time = None
            target_ts = None
            p_world = None
            p_cam = None
            uv = None

            if projection_ready:
                global_time = global_map.get(capture_aria_idx)
                if global_time is None and g0 is not None and g1 is not None:
                    ratio = frame_idx / max(
                        1.0, float(timesync_end_idx - timesync_start_idx)
                    )
                    global_time = g0 + ratio * (g1 - g0)

                if global_time is not None and g0 is not None and g1 is not None:
                    alpha = (global_time - g0) / max(1e-9, (g1 - g0))
                    target_ts = ts0 + alpha * (ts1 - ts0)

                    j = int(np.searchsorted(traj_ts_arr, target_ts))
                    if j <= 0:
                        p_world = traj_xyz_arr[0]
                    elif j >= len(traj_ts_arr):
                        p_world = traj_xyz_arr[-1]
                    else:
                        t1 = traj_ts_arr[j - 1]
                        t2 = traj_ts_arr[j]
                        w = (target_ts - t1) / max(1e-9, (t2 - t1))
                        p_world = (1.0 - w) * traj_xyz_arr[j - 1] + w * traj_xyz_arr[j]

                    p_cam = r_cam_world.dot(p_world - t_world_cam)
                    uv = kb3_project(p_cam, intr)

            exo_marked = exo_frame.copy()
            h_exo, w_exo = exo_marked.shape[:2]
            in_front = bool(p_cam is not None and float(p_cam[2]) > 1e-9)
            in_image = False
            u = None
            v = None
            if uv is not None:
                u, v = float(uv[0]), float(uv[1])
                in_image = 0.0 <= u < w_exo and 0.0 <= v < h_exo
                if in_image:
                    center = (int(round(u)), int(round(v)))
                    cv2.circle(
                        exo_marked,
                        center,
                        args.marker_radius,
                        (0, 255, 0),
                        args.marker_thickness,
                    )
                    cv2.drawMarker(
                        exo_marked,
                        center,
                        (0, 255, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=args.marker_cross_size,
                        thickness=args.marker_thickness,
                    )

            ego_name = f"ego_{rank}.jpg"
            exo_name = f"exo_{rank}.jpg"
            exo_marked_name = f"exo_marked_{rank}.jpg"
            cv2.imwrite(
                str(take_out / ego_name), ego_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 98]
            )
            cv2.imwrite(
                str(take_out / exo_name), exo_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98]
            )
            cv2.imwrite(
                str(take_out / exo_marked_name),
                exo_marked,
                [int(cv2.IMWRITE_JPEG_QUALITY), 98],
            )

            frame_meta.append(
                {
                    "rank": rank,
                    "frame_idx": frame_idx,
                    "time_sec_take_video": frame_idx / fps,
                    "capture_aria_idx": capture_aria_idx,
                    "global_time": global_time,
                    "tracking_timestamp_us": target_ts,
                    "laplacian_score": sf.laplacian_score,
                    "projection_available": projection_ready,
                    "projection_reason": projection_reason,
                    "projection": {
                        "u": u,
                        "v": v,
                        "in_front": in_front,
                        "in_image": in_image,
                    },
                    "files": {
                        "ego": ego_name,
                        "exo": exo_name,
                        "exo_marked": exo_marked_name,
                    },
                }
            )

        metadata = {
            "take_name": take_name,
            "take_uid": take.get("take_uid"),
            "capture_uid": take.get("capture_uid"),
            "capture_name": capture_name,
            "best_exo": exo_cam,
            "selection": {
                "top_k": args.top_k,
                "sample_step": args.sample_step,
                "min_gap": args.min_gap,
                "score": "laplacian_on_rectified_ego_center",
                "diversity": "cosine_novelty_greedy",
                "ego_crop_ratio": args.crop_ratio,
                "ego_rectify_mode": args.ego_rectify_mode,
            },
            "projection": {
                "model": "KANNALABRANDTK3",
                "time_mapping": "capture_timesync_aria_global_to_take_trajectory_linear",
                "world_point": "device_center_from_closed_loop_trajectory",
            },
            "ego_rectification": {
                "mode": args.ego_rectify_mode,
                "source": "projectaria_tools_calibration"
                if args.ego_rectify_mode == "official"
                else "equidistant_approximation",
                "camera_label": ARIA_RGB_LABEL
                if args.ego_rectify_mode == "official"
                else None,
            },
            "frames": frame_meta,
        }

        with open(take_out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        with open(take_out / "metadata.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "frame_idx",
                    "time_sec_take_video",
                    "laplacian_score",
                    "u",
                    "v",
                    "in_front",
                    "in_image",
                    "ego",
                    "exo",
                    "exo_marked",
                ]
            )
            for fm in frame_meta:
                proj_obj = fm.get("projection", {})
                files_obj = fm.get("files", {})
                proj = proj_obj if isinstance(proj_obj, dict) else {}
                files = files_obj if isinstance(files_obj, dict) else {}
                writer.writerow(
                    [
                        fm.get("rank"),
                        fm.get("frame_idx"),
                        fm.get("time_sec_take_video"),
                        fm.get("laplacian_score"),
                        proj.get("u"),
                        proj.get("v"),
                        proj.get("in_front"),
                        proj.get("in_image"),
                        files.get("ego"),
                        files.get("exo"),
                        files.get("exo_marked"),
                    ]
                )

        summary.append(
            {
                "take_name": take_name,
                "output_dir": str(take_out),
                "selected_frames": [x.frame_idx for x in selected],
                "best_exo": exo_cam,
            }
        )
        print(f"[ok] {take_name} -> {take_out}")

    summary_name = "summary.json"
    if args.num_shards > 1:
        summary_name = f"summary.shard-{args.shard_index}-of-{args.num_shards}.json"
    with open(out_dir / summary_name, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(summary)} take packages to {out_dir / summary_name}")


if __name__ == "__main__":
    main()
