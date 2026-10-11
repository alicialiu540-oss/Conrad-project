#!/bin/zsh

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Project environment is missing."
  echo "Press Return to close this window."
  read
  exit 1
fi

STANDARD_DIR="$PROJECT_DIR/data/inbox/leg_rehab_20"
STANDARD_VIDEO=$(osascript -e "POSIX path of (choose file with prompt \"Choose the standard rehabilitation video for the same exercise\" default location POSIX file \"$STANDARD_DIR\" of type {\"mp4\", \"mov\", \"m4v\"})")
if [[ $? -ne 0 || -z "$STANDARD_VIDEO" ]]; then
  exit 0
fi

USER_VIDEO=$(osascript -e 'POSIX path of (choose file with prompt "Choose your own recorded video" of type {"mp4", "mov", "m4v"})')
if [[ $? -ne 0 || -z "$USER_VIDEO" ]]; then
  exit 0
fi

OUTPUT_DIR="$PROJECT_DIR/outputs/comparisons/latest_comparison"
echo "Analyzing and comparing both videos. Please wait..."
echo
".venv/bin/python" "dataset_tools/compare_rehab_videos.py" \
  --standard "$STANDARD_VIDEO" \
  --user "$USER_VIDEO" \
  --output-dir "$OUTPUT_DIR"
STATUS=$?

echo
if [[ $STATUS -eq 0 ]]; then
  echo "Comparison complete."
  open "$OUTPUT_DIR"
  open "$OUTPUT_DIR/comparison_video.mp4"
else
  echo "Comparison failed. Read the error message above."
fi
echo "Press Return to close this window."
read
exit $STATUS
