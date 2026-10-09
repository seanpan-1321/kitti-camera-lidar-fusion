# Camera × LiDAR Object Detection with Distance Estimation (KITTI)

> Project roadmap + step-by-step guide.
> Owner: Sean Pan (Tokyo Denki University, Information Systems Design, 3rd year)
> Purpose: portfolio project for the TIER IV 2028 new-grad engineer application (target role: LiDAR/camera object recognition)
> Hard deadline for the core project: **Sun Nov 1, 2026**. ES submission: around Nov 27, 2026.

---

## 0. How to use this file (for Sean)

1. Create an empty GitHub repo named `kitti-camera-lidar-fusion` (Public, with a Python `.gitignore`, no README).
2. Clone it on your Mac and put this file in the repo root as **`PLAN.md`**.
3. Open Claude Code in that folder and say:
   > "Read PLAN.md. We are starting Phase 0. Explain what we'll do first, then guide me step by step."
4. Work one phase at a time. At the end of each session, ask Claude Code to update `results/log.md` and commit.

---

## 1. Project summary

**One-line pitch:** YOLO finds *what* and *where* an object is in the camera image, and LiDAR points that fall inside each box tell *how far away* it is.

**Pipeline**

```
camera image ──► YOLO (2D detection) ──► boxes: [class, x1, y1, x2, y2, conf]
                                                        │
LiDAR point cloud ──► project to image pixels ──► points inside each box
  (calibration matrices)                                │
                                                        ▼
                                       distance per object (e.g. "Car 14.2 m")
                                                        │
                                                        ▼
                         evaluate vs. KITTI ground-truth 3D positions
```

**Why this project (mapping to the TIER IV listing)**

| Listing item | How this project covers it |
|---|---|
| Example role: object recognition using LiDAR, camera, etc. (LiDAR・カメラ…物体認識) | Camera detection + LiDAR fusion on an autonomous-driving dataset |
| Required: research or project experience with sensor- or algorithm-related systems | Calibration, projection, evaluation pipeline |
| Preferred: presentation and sharing technical knowledge | Japanese article on Zenn/Qiita (Phase 8) |
| Preferred: bilingual English/Japanese communication | English README + Japanese article |
| Ideal candidate: enjoys challenges and learns on his own | Failure analysis → hypothesis → measured fix |

---

## 2. Instructions for Claude Code (working agreement)

Claude Code: please follow these rules throughout the project.

- **Sean is learning.** He knows Python and PyTorch (he has built image classifiers with EfficientNet-B0 and ResNet18) but is new to object detection, LiDAR, and camera geometry.
  - Before writing code for a new concept, explain it **intuition first, then math**, using a concrete numeric example.
  - Explain every symbol in a formula.
  - Keep explanations short and use bullet points.
- **Go step by step.** Finish one step, have Sean run it and confirm the output, then move on. Don't generate the whole project at once.
- **Never invent results.** All metrics in the README and log must come from actual runs. Leave placeholders (`TBD`) until a real number exists.
- **Keep the experiment log.** After every training or eval run, append an entry to `results/log.md` (template in §8).
- **Change one variable per experiment** and compare against the baseline.
- **Ask before large downloads** (anything over 1 GB) or long GPU runs.
- **Compute setup:**
  - Sean's Mac is for code, small-sample debugging, and visualization.
  - **GPU training runs on Kaggle Notebooks** (free GPU, ~30 h/week).
  - Write code so the same scripts run in both places, with paths passed as CLI args.
- **Code style:** small scripts in `scripts/` and reusable functions in `src/`, with type hints and docstrings. No giant notebooks; notebooks only call the scripts.
- **Commits:** small and descriptive, e.g. `feat: project LiDAR points onto image`.

---

## 3. Tech decisions

| Item | Choice | Reason |
|---|---|---|
| Dataset | KITTI 3D Object Detection benchmark (training split: 7,481 labeled frames) | Has camera images, LiDAR, calibration, and 3D labels in one set |
| Detector | Ultralytics YOLO (`yolo11n` → maybe `yolo11s`) | Fastest path to a strong baseline |
| Classes | `Car` (+`Van`), `Pedestrian` (+`Person_sitting`), `Cyclist` → 3 classes | Standard KITTI evaluation classes; skip `DontCare`, `Misc`, `Truck`, `Tram` at first |
| Split | 80/20 train/val by frame ID, `seed=42` | KITTI has no official val split |
| Fusion | Project LiDAR to image → points inside box → robust depth statistic | Simple, explainable, measurable |
| Env | Python 3.11, `ultralytics`, `numpy`, `opencv-python`, `matplotlib`, `pandas`, `tqdm` | |
| License note | Ultralytics is AGPL-3.0 and KITTI is CC BY-NC-SA 3.0, so the repo must be public and non-commercial | Fine for a portfolio |

---

## 4. Repo structure

```
kitti-camera-lidar-fusion/
├── PLAN.md                     # this file
├── README.md                   # written in Phase 7
├── requirements.txt
├── configs/
│   └── kitti.yaml              # YOLO dataset config
├── src/
│   ├── kitti_io.py             # read labels, calib, velodyne .bin
│   ├── projection.py           # LiDAR → image projection
│   └── fusion.py               # box + points → distance
├── scripts/
│   ├── make_split.py           # train/val split
│   ├── convert_kitti_to_yolo.py
│   ├── visualize_projection.py
│   ├── estimate_distance.py
│   └── eval_distance.py
├── notebooks/
│   └── kaggle_train.ipynb      # thin wrapper: clone repo, run scripts
├── results/
│   ├── log.md                  # experiment log
│   ├── figures/                # PNGs/GIF for README
│   └── metrics/                # CSV/JSON outputs
└── data/                       # gitignored
```

`.gitignore` must include `data/`, `runs/`, `*.pt`, `*.bin`, `.venv/`.

---

## 5. Roadmap overview

| Phase | Dates | Goal | Done when… |
|---|---|---|---|
| 0. Setup | Oct 9–12 | Repo, env, Kaggle, data access | Can open one KITTI image + its label + its LiDAR file |
| 1. Data → YOLO format | Oct 9–12 | Converted labels + split | `kitti.yaml` works; 5 random images show correct boxes |
| 2. Baseline training | Oct 13–19 | First trained detector | mAP50 / mAP50-95 per class logged |
| 3. Detection analysis | Oct 13–19 | Understand failures | 1 figure of typical failures + 3 bullet observations |
| 4. LiDAR projection | Oct 20–26 | Points drawn on image, colored by depth | Visualization looks aligned (points sit on cars) |
| 5. Distance estimation + eval | Oct 20–26 | Distance per box + error table | Error by range bucket in CSV + table |
| 6. One improvement | Oct 27–Nov 1 | One measured fix | Before/after numbers in log |
| 7. README + demo GIF | Oct 27–Nov 1 | Portfolio-ready repo | README complete; **HARD STOP Nov 1** |
| — | Nov 2–8 | G検定 (Nov 7). No project work | — |
| 8. Japanese article | Nov 9–22 | Zenn/Qiita post | Published; link in README |

If behind schedule, cut in this order: Phase 6 → Phase 3 figure polish. **Never cut Phase 5**, because the distance evaluation is the core.

---

## 6. Step-by-step

### Phase 0 — Setup

**0.1 Accounts**
- GitHub: create the repo (see §0).
- Kaggle: create an account and **verify your phone number**, which is required to use GPU and internet in notebooks.
- KITTI: register at the official KITTI site (cvlibs.net, "3D Object" benchmark). Download links come after registration.

**0.2 Local environment (Mac)**
```bash
cd kitti-camera-lidar-fusion
python3.11 -m venv .venv
source .venv/bin/activate
pip install ultralytics numpy opencv-python matplotlib pandas tqdm
pip freeze > requirements.txt
```

**0.3 Get the data.** KITTI 3D Object has four parts:

| File | Size (approx.) | Needed for |
|---|---|---|
| `data_object_image_2` (left color images) | ~12 GB | Detection |
| `data_object_label_2` (labels) | ~5 MB | Detection + GT distance |
| `data_object_calib` (calibration) | ~16 MB | Projection |
| `data_object_velodyne` (LiDAR) | ~29 GB | Fusion |

- **Plan A:** on Kaggle, search Datasets for "KITTI 3D object detection" and pick one that includes all four folders (`image_2`, `label_2`, `calib`, `velodyne`). Attach it to the notebook. No download needed.
- **Plan B:** use the official links inside a Kaggle notebook with internet ON (`wget` → `/kaggle/working`).
- **On the Mac:** keep only a small dev sample of ~50 frames with all four file types in `data/kitti_sample/`. Don't download 40 GB to the laptop.

**0.4 Sanity check (first real code).** Claude Code writes `src/kitti_io.py` with:
- `read_label(path)` → list of objects. Each line has 15 fields: `type, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y`.
  - `x1..y2` = 2D box in pixels.
  - `x, y, z` = 3D position in camera coordinates, in meters; `z` = forward distance.
- `read_calib(path)` → dict with `P2` (3×4), `R0_rect` (3×3), `Tr_velo_to_cam` (3×4).
- `read_velodyne(path)` → `np.fromfile(path, dtype=np.float32).reshape(-1, 4)` → columns `x, y, z, reflectance`.

✅ Done when: for frame `000008`, you can print its labels, the calib matrix shapes, and the number of LiDAR points (~100k+).

---

### Phase 1 — Convert to YOLO format

**Concept (explain to Sean first):**
- KITTI box = pixel corners `(x1, y1, x2, y2)`.
- YOLO box = `(class, cx, cy, w, h)`, all divided by the image size, so values fall between 0 and 1.
- Example: image 1242×375, box x 100→300, y 150→250 → `cx = 200/1242 = 0.161`, `cy = 200/375 = 0.533`, `w = 200/1242 = 0.161`, `h = 100/375 = 0.267`.
- KITTI image sizes vary slightly between frames, so read each image's real size.

**Steps**
1. `scripts/make_split.py` writes `data/splits/train.txt` and `val.txt` (frame IDs, 80/20, seed 42).
2. `scripts/convert_kitti_to_yolo.py`:
   - Class map: `Car, Van → 0`, `Pedestrian, Person_sitting → 1`, `Cyclist → 2`; skip everything else.
   - Output layout Ultralytics expects: `images/{train,val}/xxxxxx.png` (symlinks OK) and `labels/{train,val}/xxxxxx.txt`.
3. `configs/kitti.yaml`:
   ```yaml
   path: /kaggle/working/kitti_yolo   # override locally
   train: images/train
   val: images/val
   names: {0: Car, 1: Pedestrian, 2: Cyclist}
   ```
4. Visual check: draw the converted YOLO boxes back onto 5 random images and save them to `results/figures/label_check_*.png`.

✅ Done when: the boxes in the check images sit exactly on the objects.

---

### Phase 2 — Baseline training (Kaggle GPU)

**Kaggle notebook flow** (`notebooks/kaggle_train.ipynb`):
```python
!git clone https://github.com/<your-username>/kitti-camera-lidar-fusion.git
%cd kitti-camera-lidar-fusion
!pip install -q ultralytics
!python scripts/make_split.py --kitti_root /kaggle/input/<dataset>/training
!python scripts/convert_kitti_to_yolo.py --kitti_root /kaggle/input/<dataset>/training --out /kaggle/working/kitti_yolo
!yolo detect train data=configs/kitti.yaml model=yolo11n.pt imgsz=640 epochs=50 batch=32 seed=42 project=/kaggle/working/runs name=baseline
```

- Settings: GPU accelerator ON; internet ON (for pip / weights).
- Kaggle sessions stop after ~12 h, and everything in `/kaggle/working` is lost unless you save the notebook version. Download `best.pt` + `results.csv` after each run.
- Start with a **quick run first** (`epochs=3`) to confirm everything works before the 50-epoch run.

**Record in `results/log.md`:** mAP50 and mAP50-95 overall and per class (from `yolo detect val`), training time, GPU type.

✅ Done when: baseline numbers are logged and `best.pt` is saved locally (not committed; it's gitignored).

---

### Phase 3 — Detection analysis

1. Run predictions on ~50 val images and save the images with drawn boxes.
2. Sort the failures into 3–4 buckets, e.g.:
   - small or far objects
   - occluded objects
   - pedestrian vs. cyclist confusion
   - false positives on poles or signs
3. Use KITTI's `occluded` field (0–3) and box height to check whether recall drops for occluded or small objects.
4. Save one grid figure of typical failures to `results/figures/failures.png`.

✅ Done when: 3 bullet observations, each backed by a number or figure, are in the log.

---

### Phase 4 — Project LiDAR onto the image

**Concept (explain intuitively first):**
- The LiDAR and the camera sit at different positions on the car and look in different directions.
- To ask "which pixel would this laser point appear at?", we:
  1. move the point from **LiDAR coordinates** to **camera coordinates** (`Tr_velo_to_cam`),
  2. apply a small rotation correction (`R0_rect`),
  3. project 3D → 2D like a pinhole camera (`P2`).

**Math** (each matrix padded to 4×4 where needed; `X` is a homogeneous point `[x, y, z, 1]`):

```
X_cam  = R0_rect · Tr_velo_to_cam · X_velo        # 3D point in camera coords (meters)
p      = P2 · X_cam                                # [u·d, v·d, d]
u, v   = p[0]/p[2], p[1]/p[2]                     # pixel coords
depth  = X_cam.z                                   # forward distance in meters
```

- Keep only points with `depth > 0` (in front of the camera) and `0 ≤ u < W`, `0 ≤ v < H`.
- Numeric sanity check: a point 10 m ahead of the camera on its center line should land near the image center (`u ≈ 621` for width 1242).

**Steps**
1. `src/projection.py` → `project_velo_to_image(points, calib, img_shape) -> (uv, depth)`.
2. `scripts/visualize_projection.py`: draw the points on the image, colored by depth (near = red, far = blue), and save to `results/figures/projection_*.png`.

✅ Done when: the points clearly sit on cars, road, and walls. If they look shifted or mirrored, the matrix order or padding is wrong; debug before continuing.

---

### Phase 5 — Distance estimation + evaluation (core of the project)

**Method v1**
- For each 2D box, collect the projected points with `(u, v)` inside the box.
- Estimated distance = **median** depth of those points.
- If fewer than 5 points are inside, return "no estimate" and count these cases.

**Evaluation design (two settings; explain why to Sean)**

| Setting | Boxes used | What it measures |
|---|---|---|
| A. Oracle boxes | Ground-truth 2D boxes | Error of the **fusion method alone** |
| B. Predicted boxes | YOLO boxes matched to GT with IoU ≥ 0.5 | **End-to-end** system error |

- Ground-truth distance = the label's `z` (forward depth of the 3D box center, in meters).
- Metrics, by range bucket (0–20 m, 20–40 m, 40 m+) and by class:
  - mean absolute error (m)
  - median error (m)
  - % of objects with no estimate
- Output: `results/metrics/distance_eval_*.csv` + a Markdown table for the README.

**Expected insight (verify with real data; don't assume):**
- LiDAR hits the object's **near surface**, while the label `z` is the **box center**.
- So estimates should be biased short by roughly half the object's length (about 2 m for a car seen from behind).
- Check the signed error, not just the absolute error.

✅ Done when: both tables (A and B) exist with real numbers, plus 2–3 observations in the log.

---

### Phase 6 — One improvement (pick ONE, based on what Phase 3/5 showed)

| Problem observed | Candidate fix |
|---|---|
| Background points inside the box (e.g. a wall behind a pedestrian) inflate the distance | Use a low percentile (e.g. 20th) instead of the median, or keep only the largest/nearest depth cluster |
| Near-surface bias | Add half of a class-typical object length to the estimate |
| Poor detection of small/far objects | Retrain with `imgsz=1280` (KITTI images are wide, so 640 shrinks small objects) |

Run it, then compare before vs. after in the same table format. Write the result in the log **even if it didn't help**; a negative result with a reason is still a good interview story.

---

### Phase 7 — README + demo GIF (HARD STOP Nov 1)

**README outline** (English):
1. Title + one-line pitch + demo GIF (detections with distance labels over ~10 consecutive frames)
2. Motivation (why camera + LiDAR; link to autonomous driving perception)
3. Pipeline diagram
4. Results:
   - detection mAP table
   - distance error tables (A and B)
   - before/after of the improvement
5. Failure analysis (figure + bullets)
6. How to run (setup, data, commands)
7. Limitations:
   - random frame split may leak similar scenes
   - 2D-only detection
   - no temporal tracking
8. Next steps:
   - compare with Autoware's `image_projection_based_fusion` package
   - ROS 2 integration
   - 3D detection (PointPillars)
9. References (KITTI paper, Ultralytics) + licenses

Add 3–4 bullet points in Japanese at the top (日本語概要) for Japanese reviewers.

✅ Done when: someone who has never seen the repo can understand what it does in 30 seconds and reproduce it from the README.

---

### Phase 8 — Japanese article (after G検定, Nov 9–22)

- Platform: Zenn or Qiita.
- Working title: 「KITTIでカメラ×LiDARの距離推定をやってみた」 ("I tried camera × LiDAR distance estimation on KITTI").
- Content: the projection explanation (with figure), results, failures, and what you learned.
- Link it from the README and the ES.

---

## 7. Common pitfalls

- **Matrix padding:** `R0_rect` (3×3) and `Tr_velo_to_cam` (3×4) must be expanded to 4×4 before multiplying.
- **Points behind the camera:** filter `depth > 0` *before* dividing by depth.
- **Camera index:** use `image_2` with `P2` (left color camera), not `P0`.
- **YOLO labels:** an image with no kept objects still needs an empty `.txt` file.
- **Kaggle:** `/kaggle/input` is read-only, so write everything to `/kaggle/working`. Save the notebook version before closing.
- **Mac:** don't train on the Mac (too slow). Mac = code + debug on 50 frames.
- **Scope creep:** no ROS 2, no 3D detection, no tracking before Nov 1.

---

## 8. Experiment log template (`results/log.md`)

```markdown
## YYYY-MM-DD — <experiment name>
- Change vs. baseline: <one thing>
- Settings: model=..., imgsz=..., epochs=..., seed=42, GPU=...
- Result: mAP50=..., mAP50-95=... / distance MAE (0–20/20–40/40+ m) = ...
- Observation: ...
- Next: ...
```

---

## 9. What goes on the ES (only after real results exist)

- One line: what was built (camera + LiDAR fusion on KITTI).
- One number: e.g. distance error in the 0–20 m range (real value only).
- One story: failure found → hypothesis → fix → measured result.
- Links: GitHub repo (+ Zenn/Qiita article if published).
