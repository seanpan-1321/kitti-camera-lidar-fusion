"""Draw converted YOLO labels back onto random images to verify the conversion."""
import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.yolo_format import CLASS_NAMES, yolo_to_pixel_box  # noqa: E402

COLORS = {0: (0, 200, 0), 1: (0, 140, 255), 2: (255, 80, 80)}  # BGR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="YOLO dataset folder (has images/ and labels/)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = Path(args.dataset)
    images = sorted((ds / "images" / args.split).glob("*.png"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(random.Random(args.seed).sample(images, args.n)):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        label_path = ds / "labels" / args.split / f"{img_path.stem}.txt"
        for line in label_path.read_text().splitlines():
            cls, *vals = line.split()
            x1, y1, x2, y2 = yolo_to_pixel_box(*map(float, vals), w, h)
            color = COLORS[int(cls)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, CLASS_NAMES[int(cls)], (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        dst = out / f"label_check_{i}_{img_path.stem}.png"
        cv2.imwrite(str(dst), img)
        print(f"saved {dst}")


if __name__ == "__main__":
    main()
