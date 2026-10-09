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
