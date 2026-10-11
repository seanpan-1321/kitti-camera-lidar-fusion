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
    parser.add_argument("--out", required=True, help="output .zip path (with --chunk_size, _part0, _part1... is added)")
    parser.add_argument("--chunk_size", type=int, default=0, help="frames per zip; 0 = one single zip")
    args = parser.parse_args()

    root = Path(args.kitti_root)
    ids = sorted(Path(args.split_file).read_text().split())
    chosen = sorted(random.Random(args.seed).sample(ids, min(args.n, len(ids))))

    chunk = args.chunk_size or len(chosen)
    parts = [chosen[i:i + chunk] for i in range(0, len(chosen), chunk)]
    for i, part in enumerate(parts):
        out = Path(args.out)
        if len(parts) > 1:
            out = out.with_name(f"{out.stem}_part{i}{out.suffix}")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
            for fid in part:
                for d in DIRS:
                    for p in (root / d).glob(f"{fid}.*"):
                        z.write(p, f"{d}/{p.name}")
        print(f"{len(part)} frames -> {out} ({out.stat().st_size / 1e6:.0f} MB)")

if __name__ == "__main__":
    main()
