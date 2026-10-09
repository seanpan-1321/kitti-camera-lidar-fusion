"""KITTI -> YOLO label conversion helpers."""

# KITTI type -> YOLO class id. Types not listed here (DontCare, Misc, Truck, Tram) are skipped.
CLASS_MAP: dict[str, int] = {
    "Car": 0,
    "Van": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}
CLASS_NAMES: dict[int, str] = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def kitti_box_to_yolo(
    box: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    """Convert a pixel box (x1, y1, x2, y2) to normalized (cx, cy, w, h).

    The box is clipped to the image first. Returns None if nothing is left.
    """
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(img_w), x2)
    y1, y2 = max(0.0, y1), min(float(img_h), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        (x1 + x2) / 2 / img_w,
        (y1 + y2) / 2 / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    )


def yolo_to_pixel_box(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Inverse of kitti_box_to_yolo, used to draw labels back onto images."""
    return (
        round((cx - w / 2) * img_w),
        round((cy - h / 2) * img_h),
        round((cx + w / 2) * img_w),
        round((cy + h / 2) * img_h),
    )
