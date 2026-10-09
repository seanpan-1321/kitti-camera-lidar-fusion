"""Evaluate LiDAR-based distance estimates against KITTI ground-truth z.

Setting A (this script, default): ground-truth 2D boxes, so only the fusion method is measured.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion import estimate_distance  # noqa: E402
from src.kitti_io import read_calib, read_label, read_velodyne  # noqa: E402
from src.projection import project_velo_to_image  # noqa: E402
from src.yolo_format import CLASS_MAP, CLASS_NAMES  # noqa: E402

IMG_SHAPE = (375, 1242)  # (h, w); KITTI frames differ by a few pixels, which doesn't matter here
BUCKETS = [(0, 20, "0-20 m"), (20, 40, "20-40 m"), (40, np.inf, "40+ m")]


def bucket_of(z: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= z < hi:
            return name
    return "n/a"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (class, range bucket) error statistics, plus an 'all' row per class and overall."""
    def stats(g: pd.DataFrame) -> pd.Series:
        ok = g.dropna(subset=["est_z"])
        err = ok["signed_error"]
        return pd.Series({
            "n": len(g),
            "no_estimate_%": 100 * g["est_z"].isna().mean(),
            "MAE_m": err.abs().mean(),
            "median_abs_err_m": err.abs().median(),
            "mean_signed_err_m": err.mean(),
            "median_signed_err_m": err.median(),
        })

    rows = []
    for cls in [*CLASS_NAMES.values(), "All"]:
        sub = df if cls == "All" else df[df["class"] == cls]
        for bucket in [b[2] for b in BUCKETS] + ["All"]:
            g = sub if bucket == "All" else sub[sub["bucket"] == bucket]
            if len(g):
                rows.append({"class": cls, "range": bucket, **stats(g)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True, help="folder with label_2/calib/velodyne")
    parser.add_argument("--split_file", help="optional txt of frame IDs (e.g. val.txt); default: all frames")
    parser.add_argument("--out", default="results/metrics")
    parser.add_argument("--tag", default="A_oracle")
    args = parser.parse_args()

    root, out = Path(args.kitti_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.split_file:
        frames = Path(args.split_file).read_text().split()
    else:
        frames = sorted(p.stem for p in (root / "label_2").glob("*.txt"))

    rows = []
    for fid in tqdm(frames, desc="frames"):
        calib = read_calib(root / "calib" / f"{fid}.txt")
        uv, depth = project_velo_to_image(read_velodyne(root / "velodyne" / f"{fid}.bin"), calib, IMG_SHAPE)
        for obj in read_label(root / "label_2" / f"{fid}.txt"):
            if obj.type not in CLASS_MAP:
                continue
            est, n_pts = estimate_distance(obj.box, uv, depth)
            gt_z = obj.location[2]
            rows.append({
                "frame": fid, "type": obj.type, "class": CLASS_NAMES[CLASS_MAP[obj.type]],
                "occluded": obj.occluded, "truncated": obj.truncated,
                "gt_z": gt_z, "est_z": est, "n_points": n_pts,
                "signed_error": None if est is None else est - gt_z,  # negative = estimate too short
                "bucket": bucket_of(gt_z),
            })

    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(out / f"distance_eval_{args.tag}_objects.csv", index=False)
    summary.to_csv(out / f"distance_eval_{args.tag}.csv", index=False)
    md = summary.to_markdown(index=False, floatfmt=".2f")
    (out / f"distance_eval_{args.tag}.md").write_text(md + "\n")
    print(f"{len(frames)} frames, {len(df)} objects")
    print(md)


if __name__ == "__main__":
    main()
