"""Project LiDAR points onto the left color camera image (image_2 / P2)."""
import numpy as np


def project_velo_to_image(
    points: np.ndarray, calib: dict[str, np.ndarray], img_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points to pixel coordinates.

    Chain: X_cam = R0_rect @ Tr_velo_to_cam @ X_velo, then p = P2 @ X_cam,
    u = p[0] / p[2], v = p[1] / p[2].

    Args:
        points: (N, 3) or (N, 4) LiDAR points in the Velodyne frame (x forward, y left, z up).
        calib: dict with P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (3x4), as from read_calib.
        img_shape: (height, width) of the image.

    Returns:
        uv: (M, 2) pixel coordinates of the points that land inside the image.
        depth: (M,) forward distance in the camera frame, in meters.
    """
    h, w = img_shape[:2]

    # pad R0_rect (3x3) and Tr_velo_to_cam (3x4) to 4x4 so they can be chained
    r0 = np.eye(4)
    r0[:3, :3] = calib["R0_rect"]
    tr = np.eye(4)
    tr[:3, :4] = calib["Tr_velo_to_cam"]

    xyz1 = np.hstack([points[:, :3].astype(np.float64), np.ones((len(points), 1))])  # (N, 4)
    x_cam = (r0 @ tr @ xyz1.T)  # (4, N), camera coords in meters
    depth = x_cam[2]

    # drop points behind the camera BEFORE dividing by depth
    front = depth > 0
    x_cam, depth = x_cam[:, front], depth[front]

    p = calib["P2"] @ x_cam  # (3, M)
    u, v = p[0] / p[2], p[1] / p[2]

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.stack([u[inside], v[inside]], axis=1), depth[inside]
