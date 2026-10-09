# Experiment log

## 2026-10-09 — smoke run (pipeline check, not the baseline)
- Change vs. baseline: none; 3 epochs only, to verify conversion + training + validation work end to end
- Settings: model=yolo11n, imgsz=640, epochs=3, batch=32, seed=42, GPU=Tesla T4
- Data: train 5985 images / 30410 boxes, val 1496 images / 7582 boxes
- Result (val, 3 epochs): mAP50=0.607, mAP50-95=0.350
  - Car 0.858 / 0.581, Pedestrian 0.564 / 0.283, Cyclist 0.400 / 0.187 (mAP50 / mAP50-95)
- Training time: 0.090 h
- Observation: metrics rise every epoch; Car best, Cyclist recall lowest (0.315). Not converged, so not a baseline.
- Next: 50-epoch baseline run

## 2026-10-09 — Distance estimation v1, Setting A (oracle boxes), 50-frame dev sample
- Change vs. baseline: first run of the fusion method (median depth of LiDAR points inside the GT 2D box, min 5 points)
- Settings: GT 2D boxes from label_2, GT distance = label z, frames 000000-000049 (191 objects: 161 Car, 23 Pedestrian, 7 Cyclist). Small sample, so numbers are indicative only; rerun on the val split on Kaggle for the real table.
- Result (all classes): MAE 2.64 m, median abs error 1.42 m, mean signed error -1.79 m, no-estimate 0.5%
  - Car: MAE 2.91 m, median signed error -1.42 m. Pedestrian: MAE 0.90 m, median signed +0.22 m.
  - By range (all classes), MAE: 0-20 m 1.61, 20-40 m 3.50, 40+ m 2.77
- Observation:
  - Fully visible cars (occluded=0, n=86) have median signed error -1.18 m, i.e. estimates are short. This fits the near-surface hypothesis (LiDAR hits the car's rear surface, label z is the box center), but the offset is smaller than half a car length (~1.9 m), so the hypothesis is only partly confirmed.
  - Partly occluded cars (occluded=2, n=27) have median signed error -6.67 m and make up the worst outliers (e.g. gt 66.4 m, est 22.9 m). Hypothesis: the 2D box contains LiDAR points from the occluding nearer object, so the median picks the occluder's depth. Not yet checked on images.
  - Mean error is much worse than median error, so a few outliers dominate; median/percentile statistics are more robust here.
- Next: Setting B once best.pt is back from Kaggle; pick the Phase 6 improvement from these observations.

## 2026-10-09 — Outlier check for Setting A (looking at the 3 worst errors)
- Change vs. baseline: none; visual inspection only (scripts/visualize_outliers.py, results/figures/outlier_*.png)
- Observation: in all 3 worst cases (frames 000009, 000032, 000049; all occluded=2) the box contains points from a nearer object in front of the target, and the median lands on the nearer object.
  - 000009: 16 points, 12 on the near car (~23 m), 4 on the true car (66.4 m); estimate 22.9 m.
  - 000049: 77 points, most on a car at ~20 m, about 14 on the true car (~34 m); estimate 20.0 m.
  - 000032 (Van): 691 points over many depths; the single biggest peak IS the true distance (39 m), but nearer points are more numerous overall, so the median is 21.9 m.
- Implication: "pick the nearest cluster" would make these worse, and "pick the largest cluster" works in only one of three. A depth-only rule cannot separate target from occluder; a candidate fix is to drop points that fall inside another, nearer detection's box. Only 3 cases inspected so far.
- Next: Setting B with YOLO boxes; choose the Phase 6 improvement after seeing it.
