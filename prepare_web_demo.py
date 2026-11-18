"""Prepare the existing LSTM reviewer video for browser playback."""
from pathlib import Path
import subprocess
import imageio_ffmpeg

root = Path(__file__).resolve().parent
source = root / 'results/lstm-prototype/quadruped_IMG_6086/IMG_6086_reviewer_comparison.mp4'
output = root / 'results/web-demo'
if not source.is_file():
    raise SystemExit('Example recording is not installed. See README: Local web interface.')
output.mkdir(parents=True, exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg, '-y', '-i', str(source), '-an', '-c:v', 'libx264', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'example.mp4')], check=True)
subprocess.run([ffmpeg, '-y', '-ss', '3', '-i', str(output / 'example.mp4'),
                '-frames:v', '1', str(output / 'poster.jpg')], check=True)
