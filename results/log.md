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

## 2026-10-11 — Distance estimation v1, Setting A (oracle boxes), full val split
- Change vs. baseline: none; same method as the 50-frame run, now on all 1496 val frames (7582 objects: 6385 Car, 884 Pedestrian, 313 Cyclist). Counts match the YOLO val instance counts.
- Settings: GT 2D boxes, GT distance = label z, median depth of LiDAR points in box, min 5 points. Run on Kaggle (CPU) via notebooks/kaggle_eval.ipynb. Table: results/metrics/distance_eval_A_oracle_fullval.md
- Result (all classes): MAE 2.81 m, median abs error 1.39 m, mean signed error -1.85 m, no-estimate 0.47%. The 50-frame sample gave 2.64 / 1.42 / -1.79, so it was a fair preview.
- By range, all classes (MAE / median abs err): 0-20 m 1.60 / 0.91; 20-40 m 2.94 / 1.51; 40+ m 4.93 / 1.75.
- Observation:
  - Median abs error barely grows with range (0.91 -> 1.75 m), but MAE does (1.60 -> 4.93 m): at long range a minority of objects is very wrong. Car 40+ m: mean signed -4.54 m vs median -1.74 m.
  - Cars are biased short (median -1.33 m overall), in line with the near-surface hypothesis. Cyclists are biased LONG (mean +1.57 m, median +0.71 m; 0-20 m: +2.20 / +1.81 m), the opposite sign. Hypothesis (untested): cyclists are thin, so the box contains many background points behind them.
  - Pedestrians are nearly unbiased (median +0.10 m) with median abs error 0.20 m.
  - Occlusion outliers seen on the 50-frame sample (3 worst cases) probably explain the long-range tail, but this was not checked on the full val set.
- Next: Setting B (YOLO boxes); check the long-range tail by occlusion level on the full val objects CSV (it is in the Kaggle output, not downloaded yet).

## 2026-10-11 — Setting B (YOLO boxes, end-to-end) vs Setting A, 200 val frames
- Change vs. baseline: boxes come from YOLO (weights/best.pt, conf=0.25, imgsz=640) matched to GT with IoU>=0.5 and same class (greedy, by confidence), instead of the labeled boxes. Objects with no matching YOLO box are counted as "missed".
- Settings: 200 random val frames (data/kitti_val_sample, seed 0), 978 GT objects (830 Car, 112 Pedestrian, 36 Cyclist). Tables: results/metrics/distance_eval_{A_oracle,B_yolo}_valsample.md. NOTE: baseline per-class mAP table not logged yet (waiting for the Kaggle cell 5 output); results.csv last epoch gives mAP50=0.845, mAP50-95=0.574.
- Result:
  - A (all 978 objects): MAE 2.89 m, median abs err 1.43 m, mean signed -1.97 m.
  - B: 10.5% of GT objects missed (Car 7.2%, Pedestrian 30.4%, Cyclist 25.0%); on the rest MAE 2.54 m, median abs err 1.40 m, mean signed -1.69 m.
- Observation:
  - B's MAE looks lower than A's (2.54 vs 2.89) but this is survivorship bias: the 100 missed objects are the hard ones (A MAE on them: 6.11 m). On the same 874 objects that B matched, A gives MAE 2.52 m and B gives 2.54 m. So YOLO's boxes are as good as the labeled boxes for distance; the extra end-to-end cost shows up as missed objects, not as worse distances.
  - Missed rate by occlusion level: 0 -> 4.6%, 1 -> 9.6%, 2 -> 23.6%, 3 -> 12.5%. By box height: <25 px 13.8%, 25-40 px 11.3%, 40-60 px 11.3%, >60 px 8.6% (size matters less than occlusion).
  - Pedestrians are missed most often (30%) at conf=0.25; a lower confidence threshold would trade fewer misses for more false positives (not measured yet; false positives are not counted in this table).
- Next: check how the missed rate changes with conf; Phase 3 failure figure; choose the Phase 6 improvement (candidate: ignore points that fall inside another, nearer detection's box, aimed at the long-range / occluded error tail).

## 2026-10-11 — Phase 6: car near-surface bias correction (constant offset)
- Change vs. baseline: add a constant offset (m) to every Car distance estimate (--car_offset in scripts/eval_distance.py). Nothing else changes (same YOLO boxes, same median, same matching).
- Settings: Setting B (YOLO best.pt, conf 0.25, IoU 0.5) on the 200 val frames (data/kitti_val_sample). Offsets were fitted on the 50-frame dev sample minus the 1 frame that is also in the test frames (49 frames, 158 cars), so the test frames did not influence them:
  - physics: half of the mean car length, 3.92 / 2 = +1.96 m
  - data: minus the median signed error of the dev cars, +1.42 m
- Result (Car, 769 matched objects with an estimate; files: results/metrics/distance_eval_B_{baseline,offset142_data,offset196_physics}_valsample.*):
  - baseline +0.00 m: MAE 2.70, median abs err 1.44, median signed -1.33, within 1 m 30.6%, within 2 m 75.0%, within 5 m 86.4%
  - data +1.42 m: MAE 2.14, median abs err 0.67, median signed +0.09, within 1 m 61.9%, within 2 m 77.2%, within 5 m 88.6%
  - physics +1.96 m: MAE 2.32, median abs err 1.08, median signed +0.63, within 1 m 47.9%, within 2 m 73.9%, within 5 m 88.2%
  - All classes with the data offset: MAE 2.54 -> 2.05, median abs err 1.40 -> 0.65.
- Observation:
  - The data-fitted offset roughly halves the typical error (median abs 1.44 -> 0.67 m) and doubles the share of cars within 1 m (31% -> 62%). The physics offset (half length) overshoots: LiDAR points do not all sit on the rear face (side/roof points pull the median deeper), so the bias (~1.4 m) is smaller than half the car length (~2 m). Hypothesis, not tested.
  - The gain is concentrated at 20 m and beyond (Car median abs err: 20-40 m 1.53 -> 0.57, 40+ m 1.71 -> 0.56). At 0-20 m the constant overshoots (median signed -0.65 -> +0.77) and MAE gets slightly worse (1.46 -> 1.57). The bias grows with distance, so a single constant is not ideal. Hypothesis: at close range LiDAR also hits the car's side, which changes the median; not tested.
  - The long error tail is untouched (within 5 m only 86% -> 89%), consistent with the occlusion explanation seen earlier.
  - This corrects towards KITTI's label convention (box center). The nearest-surface distance may be the more useful quantity for driving; state this in the README.
  - Caveat: single run, one dev/test split, the offset was fit on 158 cars. Not yet repeated on the full val set.
- Next: try a distance-dependent offset (one more variable, separate experiment) or leave as future work; run the final numbers on the full 1496 val frames on Kaggle for the README table.

## 2026-10-11 — Phase 3: detection failure analysis (YOLO baseline, 200 val frames)
- Change vs. baseline: none; analysis only (scripts/analyze_detection.py -> results/metrics/detection_analysis.md, results/figures/failures.png). conf>=0.25, IoU>=0.5 for a match.
- Result: 978 GT objects; 875 found, 101 missed, 2 found with the wrong class. 159 false alarms (125 predicted Car, 26 Pedestrian, 8 Cyclist), about 0.8 per frame. Recall: Car 92.8%, Cyclist 75.0%, Pedestrian 69.6%.
- Observations:
  1. Hidden objects are missed most: recall by KITTI occlusion level is 95.4% (0, fully visible), 90.4% (1), 76.4% (2), 87.5% (3, tiny n=32). The failure grid shows missed cars tucked between parked cars.
  2. Pedestrians are the weak class: 32 of 112 were not detected at all. Examples show people in deep shadow and people half hidden behind a car. Pedestrian vs Cyclist confusion is rare (2 of 148 people); both examples shown are people walking next to a bicycle (the label says Pedestrian, YOLO says Cyclist).
  3. Size matters less than occlusion: recall is 86.2% for boxes under 25 px and 91.4% above 60 px. About 8 percentage points of spread, versus 19 points across occlusion levels.
  - False alarms: in 4 examples shown, several look like real parked cars or pedestrians that carry no label (hypothesis: part of the "false alarms" are unlabeled real objects, so precision is understated; not verified over all 159).
- Next: full-val numbers on Kaggle; decide on follow-ups (distance-dependent offset, lower-confidence threshold study).

## 2026-10-15 — Baseline detection score + full-val Setting B (all 1496 val frames)
- Change vs. baseline: none for detection; Setting B now run on the full val split, with and without the +1.42 m Car offset (the offset was fitted on 49 dev frames, not on val).
- Settings: YOLO11n baseline, imgsz=640, epochs=50, batch=32, seed=42, Kaggle T4, training time 4514 s (1.25 h, from results.csv). Distance eval: conf 0.25, IoU>=0.5, median depth, min 5 points. Tables: results/metrics/detection_val_full.md, distance_eval_B_fullval_{baseline,offset142}.md
- Result, detection (val, `yolo detect val`): all mAP50=0.850, mAP50-95=0.577 (P 0.866, R 0.767). Car 0.958 / 0.764, Pedestrian 0.771 / 0.434, Cyclist 0.819 / 0.533 (mAP50 / mAP50-95). Smoke run (3 epochs) was 0.607 / 0.350.
- Result, distance (Setting B, all classes): missed 10.26%, no-estimate 0.29%, MAE 2.34 m, median abs err 1.34 m, mean signed -1.45 m.
  - Car: MAE 2.50 -> 1.99 m, median abs err 1.40 -> 0.69 m, median signed -1.27 -> +0.15 m with the offset. All classes: MAE 2.34 -> 1.90 m, median abs err 1.34 -> 0.67 m.
  - By range (Car, median abs err, before -> after): 0-20 m 0.93 -> 0.89, 20-40 m 1.52 -> 0.59, 40+ m 1.74 -> 0.53. Car MAE at 0-20 m got slightly worse: 1.52 -> 1.64 m (median signed -0.64 -> +0.78, overshoot).
- Observation:
  - The Phase 6 result replicates on the full val set (200-frame run: Car median abs err 1.44 -> 0.67; full val: 1.40 -> 0.69). The offset helps from 20 m outwards and slightly overshoots at close range; a distance-dependent offset is still the obvious next experiment.
  - Missed objects grow with distance, strongly for pedestrians: 18.3% (0-20 m), 50.0% (20-40 m), 64.7% (40+ m); Cars 3.2% -> 5.9% -> 15.0%; Cyclists 14.8% -> 19.6% -> 53.1%. This refines the 200-frame note that size mattered less than occlusion: pooled over classes it did, but for small objects (pedestrians, cyclists) distance/size matters a lot. Pedestrian mAP50-95 (0.434) is the weakest class score.
  - Setting B's error numbers are lower than Setting A's (MAE 2.34 vs 2.81 on the same val set) only because B leaves out the ~10% hardest objects it missed (see the 200-frame comparison); not a like-for-like comparison.
  - Cyclists stay biased long (+2.04 m mean signed); the offset is applied to Cars only. Cause (background points inside the thin box) still untested.
  - Predictions made before the run held: ~10% missed (10.3% observed, pedestrians worst) and typical car error roughly halved (1.40 -> 0.69).
- Next: code walkthrough, then demo GIF and README. Optional follow-ups: distance-dependent offset; confidence threshold study.
