# YOLO Pose video analysis

This project uses Ultralytics YOLO Pose to detect people and estimate 17 body
keypoints in images and videos.

## MediaPipe wrist-analysis environment

MediaPipe is installed separately so its dependencies do not interfere with
YOLO. Activate it with:

```bash
cd /Users/anjunliu/ConradProject
source .venv-mediapipe/bin/activate
python -c "import mediapipe as mp; print(mp.__version__)"
```

Analyze a wrist video with:

```bash
python analyze_wrist.py "/absolute/path/to/video.mp4"
```

Results are saved under `results/mediapipe/<video-name>/` as an annotated MP4,
frame-level CSV, angle graph, and JSON summary. Use `deactivate` when finished.

The angle is a 2D image-plane estimate from elbow to wrist to the middle-finger
knuckle. Record flexion/extension from the side and radial/ulnar deviation from
above. The sign depends on camera orientation. This is not a validated clinical
measurement and should be checked against an appropriate reference method.

## MediaPipe full-body video analysis

With the MediaPipe environment activated, run:

```bash
python analyze_body.py "/absolute/path/to/video.mp4"
```

Results are saved under `results/mediapipe-body/<video-name>/`. They include an
annotated video, all 33 pose landmarks, major bilateral joint angles, a graph,
and a JSON summary. Joint angles are 2D image-plane estimates and are sensitive
to camera position, occlusion, and out-of-plane movement.

### Compare an exercise with a reference

Compare two MediaPipe `*_joint_angles.csv` files with phase alignment:

```bash
source .venv-mediapipe/bin/activate
python compare_angle_sequences.py "/path/to/movement_joint_angles.csv" "/path/to/reference_joint_angles.csv"
```

The program uses multivariate dynamic time warping to handle different video
durations, tests direct and left/right-mirrored reference alignment, and saves
per-joint deviation metrics, aligned data, graphs, and a JSON summary under
`results/angle-comparison/<movement-name>/`. This is a one-reference movement
comparison, not a validated clinical normativity score.

### LSTM movement-quality prototype

Train a one-class LSTM autoencoder on one reference recording and score another
recording with the same `*_smooth_deg` columns:

```bash
source .venv/bin/activate
python lstm_movement_quality.py "/path/to/reference_joint_angles.csv" "/path/to/movement_joint_angles.csv" --output-dir results/lstm-prototype/quadruped
```

The program filters low-coverage angles, interpolates short landmark gaps, adds
angular velocity, creates fixed-length windows, trains with early stopping, and
uses the reference validation errors to set a threshold. It saves a PyTorch
checkpoint, window and joint scores, training history, plots, and a JSON summary.

Score another processed video later without retraining:

```bash
python score_lstm_movement.py results/lstm-prototype/quadruped/lstm_autoencoder.pt "/path/to/new_joint_angles.csv" --output-csv results/lstm-prototype/new_video_scores.csv
```

Create a reviewer-facing side-by-side video using the annotated reference and
movement videos, `window_scores.csv`, and `summary.json`:

```bash
python create_lstm_review_video.py "/path/to/reference_annotated.mp4" "/path/to/movement_annotated.mp4" "/path/to/window_scores.csv" "/path/to/summary.json" "/path/to/reviewer_comparison.mp4"
```

The two videos are synchronized by normalized recording progress. The display
includes the current reconstruction error, its reference percentile, a moving
timeline, and a prototype similarity index. A value of 50 on the similarity
index corresponds to the reference-derived error threshold; it is not a
clinical grade or pass/fail boundary.

This is an engineering demonstration trained on windows from one reference
video. Its reconstruction error measures similarity to that recording; it is
not a clinical movement-quality, safety, or diagnosis score. A research model
needs multiple representative participants, repetition/quality labels, a
participant-independent test set, and clinician-led validation.

### Landmark tensor and multiclass LSTM

Create a labeling manifest by copying `landmark_dataset_manifest.example.csv`.
Each row represents one complete exercise recording and must include a sample
ID, MediaPipe landmark CSV, reviewer-assigned label, participant ID, exercise
ID, and `train`, `validation`, or `test` split.

Build a padded tensor with shape `[samples, time_steps, features]`:

```bash
source .venv/bin/activate
python prepare_landmark_lstm_dataset.py landmark_dataset_manifest.csv results/landmark-dataset.npz \
  --time-steps 128 \
  --alignment padding \
  --critical-landmarks left_knee,right_knee \
  --min-visibility 0.5 \
  --max-critical-low-visibility 0.20 \
  --include-velocity
```

Or align temporal phases to a clinician-selected reference with multivariate
dynamic time warping:

```bash
python prepare_landmark_lstm_dataset.py landmark_dataset_manifest.csv results/landmark-dataset-dtw.npz \
  --time-steps 128 \
  --alignment dtw \
  --dtw-reference "/path/to/reference_landmarks.csv" \
  --critical-landmarks left_knee,right_knee
```

Coordinates below the visibility threshold are treated as missing and
interpolated. A recording is rejected when a critical landmark exceeds the
configured low-visibility fraction. The generated JSON report records every
accepted/rejected sample and the reason. In padding mode, shorter recordings
are zero-padded and their valid lengths are stored so the LSTM ignores padding;
longer recordings are uniformly resampled so the full movement is retained.
DTW mode additionally aligns movement phases to the reference.

Train the supervised multiclass classifier:

```bash
python train_landmark_lstm_classifier.py results/landmark-dataset.npz \
  --output-dir results/landmark-lstm-classifier
```

The default label vocabulary is `Correct Execution`, `Compensatory Guarding`,
and `Reduced Range of Motion`. Labels should be assigned by qualified reviewers
using a written rubric. Do not infer an injury diagnosis from pose video alone.
Use participant-independent train, validation, and held-out test splits.

### Temporal Convolutional Network

The TCN classifier uses causal, dilated one-dimensional convolutions and
residual connections to learn patterns across time. Train it from the same
reviewed landmark tensor used by the multiclass LSTM:

```bash
source .venv/bin/activate
python train_landmark_tcn_classifier.py results/landmark-dataset.npz \
  --output-dir results/landmark-tcn-classifier \
  --channels 64,64,64,64 \
  --kernel-size 3
```

Analyze a new video over overlapping temporal windows:

```bash
python analyze_video_with_tcn.py "/absolute/path/to/video.mp4" \
  results/landmark-tcn-classifier/landmark_tcn_classifier.pt \
  --output-dir results/tcn-temporal-analysis/video-name \
  --window-frames 128 \
  --stride-frames 16
```

The video command first runs MediaPipe using `.venv-mediapipe`, then applies the
TCN using `.venv`. If landmarks have already been extracted, supply the
`*_landmarks.csv` file instead of the video. Outputs include window-level class
probabilities, a temporal probability graph, visibility-quality results, and a
JSON summary. A valid trained checkpoint requires multiple clinician-reviewed
recordings in every class; the unreviewed example tensor cannot train a useful
classifier by itself.

## Analyze a video

From Terminal:

```bash
cd /Users/anjunliu/ConradProject
source .venv/bin/activate
python analyze_pose.py /absolute/path/to/video.mp4
```

The annotated video is written to `results/pose/`.

For a different detection threshold:

```bash
python analyze_pose.py /absolute/path/to/video.mp4 --confidence 0.4
```

The included `yolo11n-pose.pt` model is the small, CPU-friendly option. This
Intel Mac has no supported GPU acceleration, so long or high-resolution videos
may take a while to process.

## Rehabilitation use

YOLO Pose estimates 2D keypoints and can miss or misplace joints when a body
part is hidden, blurred, or outside the frame. Validate any derived movement
metrics with qualified rehabilitation professionals before using them for
clinical decisions. Obtain informed consent and protect identifiable videos.

For a direct command-line run, use:

```bash
yolo pose predict model=yolo11n-pose.pt source=/absolute/path/to/video.mp4 save=True
```

The first command can take longer while the model loads. The included nano
model is suited to getting started on CPU-only Macs. Larger pose models can be
selected with `--model` when higher accuracy is worth slower processing.

### YOLO Pose angle graph

Run YOLO on an original video and export joint angles over time with:

```bash
source .venv/bin/activate
python analyze_yolo_angles.py "/absolute/path/to/original-video.mp4"
```

Results are saved under `results/yolo-angles/<video-name>/` as an annotated
video, frame-level angle CSV, graph, and JSON summary. Use the original video:
an already annotated result does not retain numerical keypoints, and its drawn
overlay may interfere with a second inference pass.

## Important rehabilitation note

The COCO pose model estimates visible 2D keypoints; it does not diagnose injury
or measure true 3D joint angles. Validate camera placement, repeatability, and
accuracy with qualified clinicians before using results for treatment decisions.

## Local web interface

Conrad now includes a responsive HTML/CSS/JavaScript interface with a Flask
backend. It provides a recorded LSTM example, video drag-and-drop uploads,
queued background analysis, annotated video playback, angle charts, measurement
tables, and CSV/JSON/video downloads.

Start it from the project directory:

```bash
.venv/bin/python -m pip install -r requirements-web.txt
.venv/bin/python prepare_web_demo.py
.venv/bin/python web_app.py
```

Open **http://127.0.0.1:5050**. Use `PORT=5051` to choose another port. The existing
`.venv` and `.venv-mediapipe` environments are used by their corresponding
analysis scripts; install the dependencies documented above if setting up a
fresh checkout. Both environments need Matplotlib for charts. The web dependencies
are installed only in `.venv`; `imageio-ffmpeg` provides the H.264 encoder used for
browser playback.

Available analysis methods:

- **Compare two videos with LSTM:** upload your movement video and a reference
  video of the same exercise. The server extracts full-body pose from both,
  trains a new LSTM autoencoder on the uploaded reference, and evaluates deviation
  in the movement recording. It never substitutes the bundled example checkpoint.
  Use at least 214 decoded reference frames and 64 movement frames (at 30 fps,
  at least 8 seconds and 3 seconds respectively), comparable camera views and
  frame rates, and visible joints. At least three common angles must have 60%
  coverage in both clips. Frame rate and exercise speed can affect scores.
  Results include a side-by-side video with a deviation timeline, window scores,
  joint deviations, training history, and the model trained for that pair.
  This remains a single-reference research comparison, not a clinical assessment.
- **Full-body joint angles:** MediaPipe landmarks, angle timeline and coverage.
- **Wrist movement:** left/right wrist angles and range of motion.
- **YOLO joint angles:** YOLO pose detection and joint-angle measurements.

The example preparation script converts
`results/lstm-prototype/quadruped_IMG_6086/IMG_6086_reviewer_comparison.mp4`
to a browser-compatible recording and poster under `results/web-demo/`. Its
metadata comes from the adjacent `summary.json`. These example assets are local
and ignored by Git; a fresh clone needs them to show the recorded demonstration.
Without them the example is marked unavailable, but uploaded-pair LSTM comparison
still works when the Python analysis environments are installed.

Uploads accept MP4/MOV/AVI, up to 250 MB per video (500 MB per pair). One job runs at a time, with up to two
additional queued jobs. Uploads and results are kept under `results/web/<job-id>/`
until removed manually while the server is stopped. They are excluded from Git.
Refresh during analysis resumes status polling within the same browser tab.
Interrupted jobs are marked failed on server restart. Pose processing has a
one-hour timeout per clip; LSTM training has a one-hour timeout and video
rendering/conversion each have a 30-minute timeout.

This server is bound to localhost and intended for a single computer. It is not
an Internet deployment: public use needs a Python/model hosting environment,
authentication, per-user access controls, storage retention, and a durable job
queue. A static HTML host alone cannot execute these Python models.

Run backend checks with:

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check web/static/app.js
```
