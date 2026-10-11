"""Phase 3: where does YOLO fail? Classify every ground-truth object and every prediction, then draw typical failures."""
import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kitti_io import read_label  # noqa: E402
from src.matching import iou, match_boxes  # noqa: E402
from src.yolo_format import CLASS_MAP, CLASS_NAMES  # noqa: E402

HEIGHT_BINS = [0, 25, 40, 60, 10_000]
HEIGHT_LABELS = ["<25 px", "25-40 px", "40-60 px", ">60 px"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="results")
    parser.add_argument("--n_examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ultralytics import YOLO

    root, out = Path(args.kitti_root), Path(args.out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    frames = sorted(p.stem for p in (root / "label_2").glob("*.txt"))

    gt_rows, fp_rows = [], []
    for fid in tqdm(frames, desc="frames"):
        labels = read_label(root / "label_2" / f"{fid}.txt")
        gts = [o for o in labels if o.type in CLASS_MAP]
        dontcare = [o.box for o in labels if o.type == "DontCare"]
        res = model.predict(str(root / "image_2" / f"{fid}.png"), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy().tolist()
        pc = res.boxes.cls.cpu().numpy().astype(int).tolist()
        pf = res.boxes.conf.cpu().numpy().tolist()

        gb, gc = [o.box for o in gts], [CLASS_MAP[o.type] for o in gts]
        matches = match_boxes(gb, gc, pb, pc, pf, 0.5)
        used = {p for p, _ in matches.values()}

        confused_preds = set()
        for i, o in enumerate(gts):
            status, pred_cls, pred_conf, best = "found", None, np.nan, np.nan
            if i in matches:
                p, best = matches[i]
                pred_cls, pred_conf = pc[p], pf[p]
            else:
                # was there a box of a DIFFERENT class on top of it?
                cand = [(iou(gb[i], pb[p]), p) for p in range(len(pb)) if p not in used and iou(gb[i], pb[p]) >= 0.5]
                if cand:
                    best, p = max(cand)
                    status, pred_cls, pred_conf = "wrong class", pc[p], pf[p]
                    confused_preds.add(p)
                else:
                    status = "missed"
            x1, y1, x2, y2 = o.box
            gt_rows.append({
                "frame": fid, "class": CLASS_NAMES[CLASS_MAP[o.type]], "type": o.type, "status": status,
                "pred_class": None if pred_cls is None else CLASS_NAMES[pred_cls], "pred_conf": pred_conf,
                "iou": best, "occluded": o.occluded, "truncated": o.truncated, "box_h": y2 - y1,
                "gt_z": o.location[2], "box": (x1, y1, x2, y2),
            })
        for p in range(len(pb)):
            if p in used or p in confused_preds:
                continue
            if any(iou(d, pb[p]) >= 0.3 for d in dontcare):
                continue  # inside a region KITTI says to ignore
            fp_rows.append({"frame": fid, "pred_class": CLASS_NAMES[pc[p]], "conf": pf[p], "box": tuple(pb[p])})

    gt, fp = pd.DataFrame(gt_rows), pd.DataFrame(fp_rows)
    gt["height_bin"] = pd.cut(gt.box_h, HEIGHT_BINS, labels=HEIGHT_LABELS)
    gt.drop(columns="box").to_csv(out / "metrics/detection_analysis_objects.csv", index=False)
    fp.drop(columns="box").to_csv(out / "metrics/detection_false_positives.csv", index=False)

    lines = [f"frames: {len(frames)}, ground-truth objects: {len(gt)}, predictions kept at conf>={args.conf}: "
             f"{int((gt.status == 'found').sum() + (gt.status == 'wrong class').sum() + len(fp))} "
             f"(false positives: {len(fp)})", ""]
    tab = pd.crosstab(gt["class"], gt.status, margins=True)
    tab["recall_%"] = 100 * tab.get("found", 0) / tab["All"]
    lines += ["Per class (status counts and recall):", tab.round(1).to_markdown(), ""]
    for col, name in [("occluded", "KITTI occlusion level"), ("height_bin", "box height")]:
        t = (gt.assign(found=gt.status == "found").groupby(col, observed=True)
             .agg(n=("found", "size"), recall_pct=("found", lambda s: 100 * s.mean())))
        lines += [f"Recall by {name}:", t.round(1).to_markdown(), ""]
    ped = gt[gt["class"].isin(["Pedestrian", "Cyclist"])]
    lines += ["Pedestrian/Cyclist outcomes (rows = truth, columns = what YOLO said):",
              pd.crosstab(ped["class"], ped.pred_class.fillna("(nothing)")).to_markdown(), ""]
    lines += ["False positives by predicted class:", fp.pred_class.value_counts().to_markdown(), ""]
    report = "\n".join(lines)
    (out / "metrics/detection_analysis.md").write_text(report + "\n")
    print(report)

    # figure: typical failures, a mix of the three kinds
    rng = random.Random(args.seed)
    n = args.n_examples
    quota = {"missed": n // 2, "wrong class": n // 4}
    picks = []
    for status, k in quota.items():
        idx = list(gt.index[gt.status == status])
        rng.shuffle(idx)
        picks += [("gt", i) for i in idx[:k]]
    fp_idx = list(fp.index)
    rng.shuffle(fp_idx)
    picks += [("fp", i) for i in fp_idx[: n - len(picks)]]

    cols = 4
    rows_n = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 2.9 * rows_n))
    for ax, (kind, i) in zip(np.atleast_1d(axes).ravel(), picks):
        r = gt.loc[i] if kind == "gt" else fp.loc[i]
        fid = r.frame
        img = cv2.cvtColor(cv2.imread(str(root / "image_2" / f"{fid}.png")), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = r.box
        pad = max(40, 0.8 * max(x2 - x1, y2 - y1))
        cx1, cx2 = int(max(0, x1 - pad)), int(min(img.shape[1], x2 + pad))
        cy1, cy2 = int(max(0, y1 - pad)), int(min(img.shape[0], y2 + pad))
        ax.imshow(img[cy1:cy2, cx1:cx2], extent=(cx1, cx2, cy2, cy1))
        gcol = "lime" if kind == "gt" else "red"
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=gcol, lw=2,
                                   ls="-" if kind == "gt" else "--"))
        if kind == "gt":
            sub = f"{r['class']} at {r.gt_z:.0f} m, occl={r.occluded}, h={r.box_h:.0f}px"
            title = f"MISSED: {sub}" if r.status == "missed" else f"WRONG CLASS (said {r.pred_class}): {sub}"
        else:
            title = f"FALSE ALARM: {r.pred_class} ({r.conf:.2f})"
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("Typical YOLO failures (green solid = real object, red dashed = false alarm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figures/failures.png", dpi=110)
    print(f"saved {out / 'figures/failures.png'}")


if __name__ == "__main__":
    main()
