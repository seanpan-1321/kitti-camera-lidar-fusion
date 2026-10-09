"""Combine a 2D box with projected LiDAR points to estimate an object's distance."""
import numpy as np

MIN_POINTS = 5


def estimate_distance(
    box: tuple[float, float, float, float],
    uv: np.ndarray,
    depth: np.ndarray,
    min_points: int = MIN_POINTS,
) -> tuple[float | None, int]:
    """Estimate distance as the median depth of the LiDAR points inside a box.

    Args:
        box: (x1, y1, x2, y2) in pixels.
        uv: (M, 2) pixel coordinates of projected LiDAR points.
        depth: (M,) depth of each point in meters.
        min_points: fewer points than this means no estimate.

    Returns:
        (distance in meters or None, number of points inside the box)
    """
    x1, y1, x2, y2 = box
    inside = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    n = int(inside.sum())
    if n < min_points:
        return None, n
    return float(np.median(depth[inside])), n
