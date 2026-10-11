"""Evaluate LiDAR-based distance estimates against KITTI ground-truth z.

Setting A (default): ground-truth 2D boxes, so only the fusion method is measured.
Setting B (--weights): YOLO boxes matched to ground truth with IoU >= 0.5, so the whole system is measured.
A ground-truth object with no matching YOLO box counts as "missed".
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
from src.matching import match_boxes  # noqa: E402
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
    """Per (class, range bucket) error statistics, plus an 'all' row per class and overall.

    missed_%       = ground-truth objects with no box to measure (Setting B: no YOLO match)
    no_estimate_%  = objects that had a box but fewer than 5 LiDAR points inside it (of all objects)
    Error statistics use only objects that got an estimate.
    """
    def stats(g: pd.DataFrame) -> pd.Series:
        err = g.dropna(subset=["est_z"])["signed_error"]
        return pd.Series({
            "n": len(g),
            "missed_%": 100 * (~g["matched"]).mean(),
            "no_estimate_%": 100 * (g["matched"] & g["est_z"].isna()).mean(),
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
    parser.add_argument("--kitti_root", required=True, help="folder with image_2/label_2/calib/velodyne")
    parser.add_argument("--split_file", help="optional txt of frame IDs (e.g. val.txt); default: all frames")
    parser.add_argument("--weights", help="YOLO weights (best.pt). If given: Setting B (YOLO boxes)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (Setting B)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--out", default="results/metrics")
    parser.add_argument("--tag", help="output name; default A_oracle or B_yolo")
    args = parser.parse_args()
    tag = args.tag or ("B_yolo" if args.weights else "A_oracle")

    root, out = Path(args.kitti_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.split_file:
        frames = Path(args.split_file).read_text().split()
    else:
        frames = sorted(p.stem for p in (root / "label_2").glob("*.txt"))
    if not frames:
        raise SystemExit("No frames to evaluate (empty split file or no labels). Check --kitti_root / --split_file.")

    model = None
    if args.weights:
        from ultralytics import YOLO  # imported here so Setting A doesn't need it

        model = YOLO(args.weights)

    rows = []
    for fid in tqdm(frames, desc="frames"):
        calib = read_calib(root / "calib" / f"{fid}.txt")
        uv, depth = project_velo_to_image(read_velodyne(root / "velodyne" / f"{fid}.bin"), calib, IMG_SHAPE)
        gts = [o for o in read_label(root / "label_2" / f"{fid}.txt") if o.type in CLASS_MAP]

        matches = {i: (None, 1.0) for i in range(len(gts))}  # Setting A: every object uses its own GT box
        pred_boxes: list = []
        if model is not None:
            res = model.predict(str(root / "image_2" / f"{fid}.png"), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            pred_boxes = res.boxes.xyxy.cpu().numpy().tolist()
            matches = match_boxes(
                [o.box for o in gts], [CLASS_MAP[o.type] for o in gts],
                pred_boxes, res.boxes.cls.cpu().numpy().astype(int).tolist(),
                res.boxes.conf.cpu().numpy().tolist(), args.iou_thr,
            )

        for i, obj in enumerate(gts):
            gt_z = obj.location[2]
            matched = i in matches
            est, n_pts, box_iou = None, 0, np.nan
            if matched:
                p_idx, box_iou = matches[i]
                box = obj.box if p_idx is None else pred_boxes[p_idx]
                est, n_pts = estimate_distance(tuple(box), uv, depth)
            rows.append({
                "frame": fid, "type": obj.type, "class": CLASS_NAMES[CLASS_MAP[obj.type]],
                "occluded": obj.occluded, "truncated": obj.truncated,
                "gt_z": gt_z, "matched": matched, "box_iou": box_iou,
                "est_z": est, "n_points": n_pts,
                "signed_error": None if est is None else est - gt_z,  # negative = estimate too short
                "bucket": bucket_of(gt_z),
            })

    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(out / f"distance_eval_{tag}_objects.csv", index=False)
    summary.to_csv(out / f"distance_eval_{tag}.csv", index=False)
    md = summary.to_markdown(index=False, floatfmt=".2f")
    (out / f"distance_eval_{tag}.md").write_text(md + "\n")
    print(f"{len(frames)} frames, {len(df)} objects")
    print(md)


if __name__ == "__main__":
    main()
