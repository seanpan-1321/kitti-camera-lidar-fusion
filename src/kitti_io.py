"""Readers for KITTI 3D object detection files: labels, calibration, LiDAR scans."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class KittiObject:
    """One labeled object. Box is in pixels; location is in camera coords (meters)."""

    type: str
    truncated: float
    occluded: int  # 0 = fully visible ... 3 = unknown
    alpha: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    dims: tuple[float, float, float]  # height, width, length
    location: tuple[float, float, float]  # x, y, z (z = forward distance)
    rotation_y: float


def read_label(path: str | Path) -> list[KittiObject]:
    """Parse a KITTI label file (one object per line, 15 fields)."""
    objects = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) < 15:
            continue
        objects.append(
            KittiObject(
                type=f[0],
                truncated=float(f[1]),
                occluded=int(float(f[2])),
                alpha=float(f[3]),
                box=tuple(float(v) for v in f[4:8]),
                dims=tuple(float(v) for v in f[8:11]),
                location=tuple(float(v) for v in f[11:14]),
                rotation_y=float(f[14]),
            )
        )
    return objects


def read_calib(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a KITTI calib file into P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (3x4)."""
    raw = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        raw[key.strip()] = np.array(values.split(), dtype=np.float64)
    return {
        "P2": raw["P2"].reshape(3, 4),
        "R0_rect": raw["R0_rect"].reshape(3, 3),
        "Tr_velo_to_cam": raw["Tr_velo_to_cam"].reshape(3, 4),
    }


def read_velodyne(path: str | Path) -> np.ndarray:
    """Load a LiDAR scan as an (N, 4) float32 array: x, y, z, reflectance."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
