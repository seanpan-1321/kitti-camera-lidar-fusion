"""Sanity check: print labels, calib shapes and LiDAR point count for one frame."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kitti_io import read_calib, read_label, read_velodyne  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True, help="folder with image_2/label_2/calib/velodyne")
    parser.add_argument("--frame", default="000008")
    args = parser.parse_args()
    root = Path(args.kitti_root)

    print(f"Frame {args.frame}")
    for obj in read_label(root / "label_2" / f"{args.frame}.txt"):
        if obj.type == "DontCare":
            continue
        print(f"  {obj.type:<10} z={obj.location[2]:6.2f} m  box={obj.box}  occluded={obj.occluded}")

    for name, mat in read_calib(root / "calib" / f"{args.frame}.txt").items():
        print(f"  {name}: shape {mat.shape}")

    points = read_velodyne(root / "velodyne" / f"{args.frame}.bin")
    print(f"  LiDAR points: {len(points)}  (columns x, y, z, reflectance)")


if __name__ == "__main__":
    main()
