"""Show the worst distance-estimate errors: image crop with the box's LiDAR points, plus a depth histogram."""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kitti_io import read_calib, read_label, read_velodyne  # noqa: E402
from src.projection import project_velo_to_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True)
    parser.add_argument("--objects_csv", default="results/metrics/distance_eval_A_oracle_objects.csv")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--max_depth", type=float, default=50.0)
    args = parser.parse_args()

    root, out = Path(args.kitti_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.objects_csv, dtype={"frame": str})
    df = df.dropna(subset=["est_z"]).assign(abs_err=lambda d: d.signed_error.abs())
    worst = df.sort_values("abs_err", ascending=False).head(args.n)

    for rank, row in enumerate(worst.itertuples(), 1):
        fid = f"{int(row.frame):06d}"
        img = cv2.cvtColor(cv2.imread(str(root / "image_2" / f"{fid}.png")), cv2.COLOR_BGR2RGB)
        calib = read_calib(root / "calib" / f"{fid}.txt")
        uv, depth = project_velo_to_image(read_velodyne(root / "velodyne" / f"{fid}.bin"), calib, img.shape)

        # find this object's box again (match by class type + gt z)
        obj = min(
            (o for o in read_label(root / "label_2" / f"{fid}.txt") if o.type == row.type),
            key=lambda o: abs(o.location[2] - row.gt_z),
        )
        x1, y1, x2, y2 = obj.box
        inside = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)

        pad = max(40, 0.4 * (x2 - x1))
        cx1, cx2 = int(max(0, x1 - pad)), int(min(img.shape[1], x2 + pad))
        cy1, cy2 = int(max(0, y1 - pad)), int(min(img.shape[0], y2 + pad))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
        ax1.imshow(img[cy1:cy2, cx1:cx2], extent=(cx1, cx2, cy2, cy1))
        sc = ax1.scatter(uv[inside, 0], uv[inside, 1], c=depth[inside], cmap="jet_r",
                         vmin=0, vmax=args.max_depth, s=10)
        ax1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=2))
        ax1.set_xlim(cx1, cx2)
        ax1.set_ylim(cy2, cy1)
        ax1.set_title(f"frame {fid}, {row.type}, occluded={row.occluded}  ({int(inside.sum())} points in box)")
        fig.colorbar(sc, ax=ax1, label="depth (m)", fraction=0.04)

        ax2.hist(depth[inside], bins=30, color="gray")
        ax2.axvline(row.gt_z, color="green", lw=2, label=f"true distance {row.gt_z:.1f} m")
        ax2.axvline(row.est_z, color="red", lw=2, ls="--", label=f"our estimate (median) {row.est_z:.1f} m")
        ax2.set_xlabel("depth of points inside the box (m)")
        ax2.set_ylabel("number of points")
        ax2.legend()
        ax2.set_title("What depths are inside the box?")

        fig.tight_layout()
        dst = out / f"outlier_{rank}_{fid}.png"
        fig.savefig(dst, dpi=110)
        plt.close(fig)
        print(f"saved {dst}: gt {row.gt_z:.1f} m, est {row.est_z:.1f} m")


if __name__ == "__main__":
    main()
