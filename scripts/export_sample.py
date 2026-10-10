"""Zip a random subset of frames (image_2, label_2, calib, velodyne) so it can be downloaded to the Mac."""
import argparse
import random
import zipfile
from pathlib import Path

DIRS = ("image_2", "label_2", "calib", "velodyne")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True, help="folder with image_2/label_2/calib/velodyne")
    parser.add_argument("--split_file", required=True, help="txt of frame IDs to sample from (e.g. val.txt)")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output .zip path")
    args = parser.parse_args()

    root = Path(args.kitti_root)
    ids = sorted(Path(args.split_file).read_text().split())
    chosen = sorted(random.Random(args.seed).sample(ids, min(args.n, len(ids))))

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for fid in chosen:
            for d in DIRS:
                for p in (root / d).glob(f"{fid}.*"):
                    z.write(p, f"{d}/{p.name}")
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"{len(chosen)} frames -> {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
