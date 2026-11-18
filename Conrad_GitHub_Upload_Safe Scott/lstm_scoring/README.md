# LSTM rehabilitation movement scoring

This folder contains the prototype that compares a home rehabilitation video with an expert reference video.

1. YOLO Pose detects body landmarks.
2. The program calculates left and right hip, knee, thigh, and shin angles.
3. A Siamese bidirectional LSTM encodes both movement sequences.
4. Cosine similarity produces a 0–100 movement score.
5. The program exports a side-by-side video, angle graph, CSV data, and report.

The bridge-to-bridge verification scored 99.5/100. A bridge-to-squat negative test scored 0.0/100.

The model is a classroom prototype. It measures visible 2D motion similarity and must not be treated as a medical diagnosis or proof that an exercise is safe.

