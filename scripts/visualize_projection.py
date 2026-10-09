"""Draw projected LiDAR points on the image, colored by depth (near = red, far = blue)."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kitti_io import read_calib, read_velodyne  # noqa: E402
from src.projection import project_velo_to_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True, help="folder with image_2/calib/velodyne")
    parser.add_argument("--frames", nargs="+", default=["000008"])
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--max_depth", type=float, default=50.0, help="depth (m) mapped to the bluest color")
    args = parser.parse_args()

    root, out = Path(args.kitti_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmap = colormaps["jet_r"]  # reversed jet: low depth = red, high depth = blue

    for fid in args.frames:
        img = cv2.imread(str(root / "image_2" / f"{fid}.png"))
        calib = read_calib(root / "calib" / f"{fid}.txt")
        points = read_velodyne(root / "velodyne" / f"{fid}.bin")
        uv, depth = project_velo_to_image(points, calib, img.shape)

        colors = (cmap(np.clip(depth / args.max_depth, 0, 1))[:, :3] * 255).astype(int)
        for (u, v), (r, g, b) in zip(uv.astype(int), colors):
            cv2.circle(img, (u, v), 1, (int(b), int(g), int(r)), -1)  # OpenCV is BGR

        dst = out / f"projection_{fid}.png"
        cv2.imwrite(str(dst), img)
        print(f"{fid}: {len(points)} points -> {len(uv)} in image, depth {depth.min():.1f}-{depth.max():.1f} m -> {dst}")


if __name__ == "__main__":
    main()
