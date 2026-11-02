# PickleBallVisualizer: progress report (2026-10-29)

Goal: a computer-vision pipeline that finds and tracks pickleball players and the ball in match video, and later maps them onto a top-down court minimap.

This report covers Days 1 to 4 of a 14-day plan. Player detection and tracking work. The ball detector is trained on images first and will be applied to video clips afterwards.

## 1. How the pipeline works

```
video clip ──► read frames ──► YOLO person detection ──► ByteTrack (IDs) ──► court-area filter ──► drawn video
                                                                                   │
                                  Roboflow image dataset ──► train YOLO ball detector (Colab) ──► (next: ball tracking, bounces)
```

| Step | What it does | Where |
|---|---|---|
| Read / write video | OpenCV `VideoCapture` / `VideoWriter`, re-encoded to H.264 so browsers can play it | `video_io.py` |
| Player detection | Pretrained YOLO11n, person class only, confidence 0.3 | `run.py` |
| Player tracking | ByteTrack through `model.track(..., persist=True)`: links boxes across frames so each player keeps one ID | `run.py` |
| Court-area filter | Keeps a person only if the **feet** (bottom-centre of the box) are inside a polygon around the court. Removes referee and crowd | `run.py` |
| Short-track filter | Drops IDs seen in fewer than 30 frames (false detections) | `run.py` |
| Overlay | Coloured ellipse under each player's feet plus an ID label | `run.py` |
| Ball detector | YOLO11n fine-tuned on a Roboflow Universe dataset, trained in Google Colab | Colab notebook |

## 2. What was done today

**Day 1: setup and video basics**
- Created the project, virtual environment and folders.
- Read and wrote video with OpenCV. Test clip: 289 frames, 30 FPS, 1302 x 740.
- Prepared the clip with ffmpeg: trimmed the first 2 s and set a constant 30 FPS.

**Day 2: player detection and tracking**
- Ran pretrained YOLO11n with ByteTrack and cached the detections.
- Added the court-area filter and the short-track filter.
- Result on the test clip: 4 players tracked for 289, 289, 289 and 286 of 289 frames, with **no ID switches**. Two false tracks (8 and 4 frames) were removed by the short-track filter.
- Added the coloured foot-ring overlay.

**Day 3: dataset audit (Google Colab)**
- Dataset: Roboflow Universe `pickleballeverything/pickleball-vision-dukcv` v1, classes `Court` and `ball`, 6,202 images at 1920 x 1080, licence CC BY 4.0.
- Median ball size is 28 x 26 px, so training uses `imgsz=960` (the ball would shrink to about 9 px at 640).
- Found **data leakage**: 522 of 823 validation images had an identical-hash twin in train, and 97% had a near-duplicate.
- Fix: hashed every image (dHash), grouped exact duplicates (3,498 groups, largest 43 images) and re-split by whole group. Exact duplicates between train and valid dropped to 0. New split: 5,453 train and 749 valid.
- Remaining limit: 518 of 749 valid images are still within 1 bit of some train image (same camera and scene, different moment), so validation scores stay somewhat optimistic. The real test will be a separately labelled clip.
- Found **mixed label types**: some label files mixed polygon rows and box rows, so Ultralytics discarded 23% of valid. Converted 2,215 polygon rows (almost all `Court`) to bounding boxes.
- Smoke test (2 epochs on 10% of the data): 0 corrupt images after the fix, about 11 s per epoch on 10% of the data on a Tesla T4.

**Day 4: ball detector training (in progress)**
- Run: YOLO11n, `imgsz=960`, 40 epochs, `patience=10`, `scale=0.3`, batch 16.
- Final metrics: _to be filled in after training._

## 3. Timing: how long does player detection and tracking take?

Measured with `run.py`. "Detect + track time" is the model loop only (detection, ByteTrack and nothing else). Reading the file and drawing and encoding the output video are listed separately.

| Clip | Length (s) | Device | Model | Detect + track (s) | ms / frame | FPS | Real-time factor | Total incl. read and render (s) | Players tracked |
|---|---|---|---|---|---|---|---|---|---|
| match.mp4 | _fill in_ | _fill in_ | yolo11n | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| clip2.mp4 | _fill in_ | _fill in_ | yolo11n | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

**Real-time factor = detect + track seconds / clip seconds.** A 10 s clip tracked in 12 s gives 1.2x (slower than real time). Below 1.0x means faster than real time. Numbers come from `results/timing_log.csv`.

## 4. Commands

Setup (local, Linux):
```bash
mkdir -p ~/pickle_viz && cd ~/pickle_viz
mkdir -p detection tracking court events render data/clips outputs cache results
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python ultralytics numpy scipy
sudo apt install -y ffmpeg
```

Prepare a clip (trim the first 2 s, constant 30 FPS, H.264):
```bash
ffmpeg -ss 2 -i inputvideos/YOUR_FILE.mp4 -vf "fps=30" -c:v libx264 -pix_fmt yuv420p -an data/clips/match.mp4
```

Run detection, tracking and timing on a clip:
```bash
python run.py --clip data/clips/match1.mp4
python run.py --clip data/clips/rally_sample.mp4      # second clip
python run.py --clip data/clips/match1.mp4 --no-roi   # different camera angle
```

Ball training (Google Colab, T4 GPU):
```python
!pip install -q roboflow ultralytics
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")   # never commit this key
dataset = rf.workspace("pickleballeverything").project("pickleball-vision-dukcv").version(1).download("yolov11")

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="/content/ds_split/data_split.yaml", imgsz=960, epochs=40,
            patience=10, batch=16, scale=0.3, save_period=5,
            project="/content/runs", name="ball_n")
```

## 5. How to tell it works on a new clip

| Check | Good sign |
|---|---|
| `players tracked` in the printout | 4 for doubles, 2 for singles |
| `frames per player ID` | Each real player close to the full frame count |
| `avg players / frame` | Close to 4 |
| Watching `outputs/<clip>_players.mp4` | Rings stay on the same player and nobody outside the court is marked |

If the number of tracked players is wrong or the referee is marked, the court-area polygon is probably wrong for that camera. Try `--no-roi`, or edit `roi` in `config.yaml`.

## 6. Limitations

- Test clips are short (about 10 s), so tracking quality on long rallies and full matches is not yet measured.
- The ball is only a few pixels wide in broadcast footage. Ball detection is the hard part of this project.
- Validation scores on the ball dataset are optimistic (see Day 3). The honest number will come from a labelled clip of our own.
- Ball height cannot be recovered from one camera, so bounce detection will rely on trajectory shape.

## 7. Credits and licences

- Ball and court dataset: "Pickleball Vision" by PickleballEverything on Roboflow Universe, CC BY 4.0.
- Models: Ultralytics YOLO11. Check the Ultralytics licence (AGPL-3.0) before any commercial use.
- Test clips are third-party footage used privately for development. They are not included in this repository.
