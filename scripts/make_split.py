"""Write train/val frame-ID lists (random split by frame, fixed seed)."""
import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True, help="folder with image_2/label_2/calib/velodyne")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ids = sorted(p.stem for p in (Path(args.kitti_root) / "label_2").glob("*.txt"))
    random.Random(args.seed).shuffle(ids)
    n_val = round(len(ids) * args.val_ratio)
    val, train = sorted(ids[:n_val]), sorted(ids[n_val:])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    print(f"{len(ids)} frames -> train {len(train)}, val {len(val)} (seed {args.seed}) in {out}")


if __name__ == "__main__":
    main()
