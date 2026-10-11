frames: 200, ground-truth objects: 978, predictions kept at conf>=0.25: 1036 (false positives: 159)

Per class (status counts and recall):
| class      |   found |   missed |   wrong class |   All |   recall_% |
|:-----------|--------:|---------:|--------------:|------:|-----------:|
| Car        |     770 |       60 |             0 |   830 |       92.8 |
| Cyclist    |      27 |        9 |             0 |    36 |       75   |
| Pedestrian |      78 |       32 |             2 |   112 |       69.6 |
| All        |     875 |      101 |             2 |   978 |       89.5 |

Recall by KITTI occlusion level:
|   occluded |   n |   recall_pct |
|-----------:|----:|-------------:|
|          0 | 454 |         95.4 |
|          1 | 272 |         90.4 |
|          2 | 220 |         76.4 |
|          3 |  32 |         87.5 |

Recall by box height:
| height_bin   |   n |   recall_pct |
|:-------------|----:|-------------:|
| <25 px       | 123 |         86.2 |
| 25-40 px     | 248 |         88.7 |
| 40-60 px     | 212 |         88.7 |
| >60 px       | 395 |         91.4 |

Pedestrian/Cyclist outcomes (rows = truth, columns = what YOLO said):
| class      |   (nothing) |   Cyclist |   Pedestrian |
|:-----------|------------:|----------:|-------------:|
| Cyclist    |           9 |        27 |            0 |
| Pedestrian |          32 |         2 |           78 |

False positives by predicted class:
| pred_class   |   count |
|:-------------|--------:|
| Car          |     125 |
| Pedestrian   |      26 |
| Cyclist      |       8 |

