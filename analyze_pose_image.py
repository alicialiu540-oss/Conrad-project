import sys
from pathlib import Path

import cv2
import mediapipe as mp

if len(sys.argv) != 2:
    raise SystemExit('Usage: python analyze_pose_image.py "/path/to/image.jpg"')

input_path = Path(sys.argv[1])
if not input_path.is_file():
    raise SystemExit(f"Image not found: {input_path}")

image = cv2.imread(str(input_path))
if image is None:
    raise SystemExit(f"Could not read image: {input_path}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5,
) as pose:
    result = pose.process(rgb)

if not result.pose_landmarks:
    raise SystemExit("No person was detected")

mp.solutions.drawing_utils.draw_landmarks(
    image,
    result.pose_landmarks,
    mp.solutions.pose.POSE_CONNECTIONS,
)

output_path = input_path.with_name(f"{input_path.stem}_pose{input_path.suffix}")
cv2.imwrite(str(output_path), image)

print(f"Annotated image saved to: {output_path}")
