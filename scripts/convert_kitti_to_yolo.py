"""Convert KITTI labels to the YOLO folder layout Ultralytics expects."""
import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kitti_io import read_label  # noqa: E402
from src.yolo_format import CLASS_MAP, kitti_box_to_yolo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True, help="folder with image_2/label_2/calib/velodyne")
    parser.add_argument("--splits", default="data/splits", help="folder with train.txt and val.txt")
    parser.add_argument("--out", required=True, help="output dataset folder")
    args = parser.parse_args()

    root, out = Path(args.kitti_root), Path(args.out)
    for split in ("train", "val"):
        ids = (Path(args.splits) / f"{split}.txt").read_text().split()
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_boxes = 0
        for fid in tqdm(ids, desc=split):
            src_img = (root / "image_2" / f"{fid}.png").resolve()
            dst_img = img_dir / f"{fid}.png"
            if not dst_img.exists():
                dst_img.symlink_to(src_img)

            with Image.open(src_img) as im:  # image size varies slightly between frames
                img_w, img_h = im.size

            lines = []
            for obj in read_label(root / "label_2" / f"{fid}.txt"):
                if obj.type not in CLASS_MAP:
                    continue
                yolo = kitti_box_to_yolo(obj.box, img_w, img_h)
                if yolo is None:
                    continue
                cx, cy, w, h = yolo
                lines.append(f"{CLASS_MAP[obj.type]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            # frames with no kept objects still get an (empty) label file
            (lbl_dir / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_boxes += len(lines)
        print(f"{split}: {len(ids)} images, {n_boxes} boxes")


if __name__ == "__main__":
    main()
