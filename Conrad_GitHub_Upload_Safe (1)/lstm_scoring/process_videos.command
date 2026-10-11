#!/bin/zsh

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "The project environment is missing. Run: python3 -m venv .venv"
  exit 1
fi

echo "Conrad YOLO Pose video importer"
echo "Put videos in: $PROJECT_DIR/data/inbox"
echo "Close the Excel tracker before processing videos."
echo

".venv/bin/python" "dataset_tools/process_videos.py"
STATUS=$?

echo
if [[ $STATUS -eq 0 ]]; then
  echo "Finished."
else
  echo "Processing stopped with an error. Read the message above."
fi
echo "Press Return to close this window."
read
exit $STATUS
