"""Match predicted 2D boxes to ground-truth 2D boxes by IoU."""
import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two pixel boxes (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    gt_boxes: list[Box],
    gt_cls: list[int],
    pred_boxes: list[Box],
    pred_cls: list[int],
    pred_conf: list[float],
    iou_thr: float = 0.5,
) -> dict[int, tuple[int, float]]:
    """Greedy one-to-one matching of predictions to ground truth.

    Predictions are visited from most to least confident. Each takes the unmatched
    ground-truth box of the same class with the highest IoU, if that IoU >= iou_thr.

    Returns:
        {gt_index: (pred_index, iou)} for matched ground-truth boxes.
    """
    matches: dict[int, tuple[int, float]] = {}
    for p in np.argsort(pred_conf)[::-1]:
        best_g, best_iou = None, iou_thr
        for g, gb in enumerate(gt_boxes):
            if g in matches or gt_cls[g] != pred_cls[p]:
                continue
            v = iou(gb, pred_boxes[p])
            if v >= best_iou:
                best_g, best_iou = g, v
        if best_g is not None:
            matches[best_g] = (int(p), best_iou)
    return matches
