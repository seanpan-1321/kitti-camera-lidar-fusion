<!-- YOLO11n baseline (50 epochs, imgsz 640, seed 42), `yolo detect val` on all 1496 val frames, pasted from the Kaggle run of notebooks/kaggle_eval.ipynb (cell 6). -->
| class      | images | instances | precision | recall | mAP50 | mAP50-95 |
|:-----------|-------:|----------:|----------:|-------:|------:|---------:|
| all        |   1496 |      7582 |     0.866 |  0.767 | 0.850 |    0.577 |
| Car        |   1372 |      6385 |     0.920 |  0.906 | 0.958 |    0.764 |
| Pedestrian |    335 |       884 |     0.849 |  0.643 | 0.771 |    0.434 |
| Cyclist    |    215 |       313 |     0.828 |  0.752 | 0.819 |    0.533 |
