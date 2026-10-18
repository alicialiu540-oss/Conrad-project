"""Run YOLO pose estimation on a video and save an annotated copy."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Ultralytics pose model name or local .pt path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for annotated output (default: results)",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")

    model = YOLO(args.model)
    model.predict(
        source=str(args.video),
        conf=args.confidence,
        save=True,
        project=str(args.output_dir),
        name="pose",
        exist_ok=True,
    )
    print(f"Annotated video saved under: {args.output_dir / 'pose'}")


if __name__ == "__main__":
    main()
