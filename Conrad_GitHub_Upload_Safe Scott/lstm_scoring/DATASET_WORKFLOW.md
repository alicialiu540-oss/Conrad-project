# Conrad YOLO Pose dataset workflow

The importer processes stretching videos and writes the technical results into the dataset tracker automatically.

## Everyday use

1. Close `Conrad_Pose_Dataset_Tracker.xlsx` if it is open.
2. Put new videos in `data/inbox/`.
3. Double-click `process_videos.command`.
4. Enter an anonymous participant ID, exercise name, camera view, and confirm permission to use each video.
5. Review the new rows in the `Videos` and `Movement Clips` sheets.

The importer skips a video it has already processed with the same model and settings.

You can reduce the questions by naming a video like this:

`P001__back_extension__side_left.mp4`

Camera view must be `front`, `side_left`, `side_right`, `back`, or `diagonal`.

## Automatic outputs

- `data/processed/keypoints/`: one CSV row per video frame, containing the primary person's 17 YOLO body points and confidence values.
- `data/processed/annotated_videos/`: a video showing the detected skeleton. This review copy has no audio.
- `data/processed/processed_videos.json`: prevents duplicate imports and records how each video was processed.
- `outputs/conrad-pose-dataset/Conrad_Pose_Dataset_Tracker.xlsx`: receives one source-video row and one full-video review row.

## Leg angle graphs

Run `dataset_tools/analyze_leg_angles.py` after YOLO processing. It creates:

- `data/processed/angle_plots/`: left/right leg-angle graphs over time.
- `data/processed/angle_data/`: frame-by-frame angle CSV files.
- `data/processed/leg_rehab_20_angle_summary.csv`: valid coverage, minimum, maximum, and movement range for every angle.

The graph includes hip angle, knee angle, thigh tilt, and shin tilt. YOLO Pose has hip, knee, and ankle points but no toe or heel points, so it cannot calculate a true ankle-joint angle. A model with foot landmarks is required for that measurement.

## Compare a standard video with your own video

Double-click `compare_rehab_videos.command`, choose a standard clip and then choose your own recording of the same exercise. The tool runs YOLO Pose when needed and creates these files in `outputs/comparisons/latest_comparison/`:

- `comparison_video.mp4`: standard and user skeleton videos side by side.
- `comparison_graph.png`: aligned knee and hip angle curves.
- `comparison_report.txt` and `.json`: similarity, movement range, timing, and observations.
- `comparison_data.csv`: aligned angle values for later research or training.

The comparison score measures visible 2D pose similarity. It is not a medical diagnosis or proof that an exercise is safe.

The primary score uses `models/leg_rehab_siamese_lstm.pt`, a Siamese LSTM that encodes the standard and user angle sequences and compares their motion embeddings. Train or refresh it with `dataset_tools/train_siamese_lstm.py`. The current model is a prototype trained from augmented pairs based on one source demonstrator; collect labelled correct and incorrect recordings from multiple participants before treating the score as reliable.

YOLO selects the largest detected person as the primary subject. Keep only one participant in frame when recording exercises.

## Human review

The importer does not decide whether a rehabilitation movement is medically correct. New movement rows use `uncertain` and `needs_review`. A reviewer must set accurate repetition timestamps, correctness, and error labels. A qualified physical therapist should approve definitions used for patient-facing feedback.

Use anonymous IDs. Do not record names, contact details, addresses, or diagnoses in the tracker.
