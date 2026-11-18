"""Local video-analysis UI. Run with .venv/bin/python web_app.py."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / 'results/lstm-prototype/quadruped_IMG_6086'
DEMO = ROOT / 'results/web-demo'
MODES = {
    'lstm': ('Compare two videos with LSTM', 'analyze_body.py', '.venv-mediapipe'),
    'body': ('Full-body movement', 'analyze_body.py', '.venv-mediapipe'),
    'wrist': ('Wrist movement', 'analyze_wrist.py', '.venv-mediapipe'),
    'yolo': ('YOLO joint angles', 'analyze_yolo_angles.py', '.venv'),
}
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config.update(MAX_CONTENT_LENGTH=501 * 1024 * 1024, JOB_ROOT=ROOT / 'results' / 'web')
executor = ThreadPoolExecutor(max_workers=1)
slots = threading.BoundedSemaphore(3)


def job_dir(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        abort(404)
    path = Path(app.config['JOB_ROOT']) / job_id
    if not (path / 'status.json').exists():
        abort(404)
    return path


def save_status(path, **data):
    temporary = path / 'status.tmp'
    temporary.write_text(json.dumps(data))
    temporary.replace(path / 'status.json')


def analyze(path, mode, filename, reference_filename=None):
    try:
        save_status(path, state='running', filename=filename, reference_filename=reference_filename, mode=mode)
        _, script, environment = MODES[mode]
        source = next(path.glob('input.*'))
        output = path / 'output'
        output.mkdir()
        command = [str(ROOT / environment / 'bin/python'), '-u', str(ROOT / script),
                   str(source), '--output-dir', str(output)]
        with (path / 'analysis.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, timeout=3600)
        if result.returncode:
            raise RuntimeError('Analysis could not finish. Check the analysis log; the video may be unreadable or a model dependency may be missing.')
        summary_file = next(output.glob('*_summary.json'))
        summary = json.loads(summary_file.read_text())
        if not summary.get('frames_processed'):
            raise RuntimeError('No video frames could be decoded. Try exporting the video as an H.264 MP4.')
        summary.pop('input_video', None)
        summary_file.write_text(json.dumps(summary, indent=2))
        comparison_video = None
        if mode == 'lstm':
            save_status(path, state='running', filename=filename, reference_filename=reference_filename,
                        mode=mode, stage='Extracting movement from your reference video')
            reference = next(path.glob('reference.*'))
            with (path / 'analysis.log').open('a') as log:
                result = subprocess.run([str(ROOT / '.venv-mediapipe/bin/python'), '-u',
                                         str(ROOT / 'analyze_body.py'), str(reference),
                                         '--output-dir', str(output)],
                                        cwd=ROOT, stdout=log, stderr=log, timeout=3600)
            if result.returncode:
                raise RuntimeError('Could not analyze the reference video. Check the log and upload a readable video with the whole body visible.')
            reference_summary_path = next(output.glob('reference*_summary.json'))
            reference_summary = json.loads(reference_summary_path.read_text())
            reference_summary.pop('input_video', None)
            reference_summary_path.write_text(json.dumps(reference_summary, indent=2))
            reference_frames = reference_summary.get('frames_processed', 0)
            if reference_frames < 214 or summary['frames_processed'] < 64:
                raise RuntimeError('LSTM comparison requires at least 214 reference frames and 64 movement frames. At 30 fps, use at least 8 seconds for the reference and 3 seconds for your movement.')
            save_status(path, state='running', filename=filename, reference_filename=reference_filename,
                        mode=mode, stage='Training the LSTM on your reference and measuring deviation')
            model_output = path / 'lstm'
            with (path / 'analysis.log').open('a') as log:
                result = subprocess.run([str(ROOT / '.venv/bin/python'), '-u',
                                         str(ROOT / 'lstm_movement_quality.py'),
                                         str(next(output.glob('reference*_joint_angles.csv'))),
                                         str(next(output.glob('input*_joint_angles.csv'))),
                                         '--output-dir', str(model_output)], cwd=ROOT,
                                        env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                                        stdout=log, stderr=log, timeout=3600)
            if result.returncode:
                raise RuntimeError('LSTM comparison failed. Both clips need at least three common joint angles visible in 60% of frames. See the analysis log for details.')
            comparison = json.loads((model_output / 'summary.json').read_text())
            comparison.pop('reference_csv', None)
            comparison.pop('movement_csv', None)
            comparison.update(reference_filename=reference_filename, movement_filename=filename)
            (model_output / 'summary.json').write_text(json.dumps(comparison, indent=2))
            for file in model_output.iterdir():
                if file.is_file():
                    shutil.copy2(file, output / ('lstm_' + file.name))
            summary['lstm'] = {
                'windows': comparison['movement_windows'],
                'mean_error': comparison['movement_mean_error'],
                'percent_above_threshold': comparison['percent_windows_above_threshold'],
                'threshold': comparison['threshold'],
                'reference_filename': reference_filename,
            }
            save_status(path, state='running', filename=filename, reference_filename=reference_filename,
                        mode=mode, stage='Preparing your side-by-side comparison video')
            comparison_video = output / 'comparison.mp4'
            with (path / 'analysis.log').open('a') as log:
                subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'create_lstm_review_video.py'),
                                str(next(output.glob('reference*annotated.mp4'))),
                                str(next(output.glob('input*annotated.mp4'))),
                                str(model_output / 'window_scores.csv'), str(model_output / 'summary.json'),
                                str(comparison_video)], cwd=ROOT, stdout=log, stderr=log,
                               check=True, timeout=1800)
        # OpenCV writes MPEG-4 Part 2; convert to H.264 for browser playback.
        preview = None
        warning = None
        try:
            import imageio_ffmpeg
            annotated = comparison_video or next(output.glob('*.mp4'))
            preview = output / 'preview.mp4'
            with (path / 'preview.log').open('w') as log:
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(annotated),
                                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                                '-movflags', '+faststart', str(preview)],
                               stdout=log, stderr=log, check=True, timeout=1800)
        except Exception:
            if preview:
                preview.unlink(missing_ok=True)
            warning = 'Browser preview unavailable. Download the annotated video to watch it locally.'
        save_status(path, state='complete', filename=filename, reference_filename=reference_filename, mode=mode, summary=summary, warning=warning)
    except subprocess.TimeoutExpired:
        save_status(path, state='error', filename=filename, reference_filename=reference_filename, mode=mode, error='Analysis exceeded the one-hour limit. Try a shorter clip.')
    except Exception as exc:
        save_status(path, state='error', filename=filename, reference_filename=reference_filename, mode=mode, error=str(exc))
    finally:
        slots.release()


@app.before_request
def local_only():
    # Refuse cross-origin uploads and DNS-rebinding hosts on this local service.
    if request.host.split(':')[0] not in {'127.0.0.1', 'localhost', '[::1]'}:
        abort(403)
    if request.method == 'POST' and request.headers.get('X-Conrad-Upload') != '1':
        abort(403)


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/modes')
def modes():
    return jsonify([{'id': key, 'name': value[0],
                     'available': (ROOT / value[2] / 'bin/python').exists() and (key != 'lstm' or (ROOT / '.venv/bin/python').exists())}
                    for key, value in MODES.items()])


@app.get('/api/example')
def example():
    summary_path = EXAMPLE / 'summary.json'
    if not summary_path.exists():
        return jsonify(available=False)
    summary = json.loads(summary_path.read_text())
    return jsonify(available=(DEMO / 'example.mp4').exists(),
                   video='/api/example/video', poster='/api/example/poster',
                   windows=summary['movement_windows'],
                   above_threshold=summary['percent_windows_above_threshold'],
                   threshold=summary['threshold'],
                   mean_error=summary['movement_mean_error'],
                   features=len(summary['angle_features']))


@app.get('/api/example/<asset>')
def example_asset(asset):
    files = {'video': 'example.mp4', 'poster': 'poster.jpg'}
    if asset not in files:
        abort(404)
    return send_from_directory(DEMO, files[asset])


@app.post('/api/jobs')
def upload():
    mode = request.form.get('mode', 'body')
    if mode not in MODES:
        return jsonify(error='Choose a supported analysis mode.'), 400
    if not (ROOT / MODES[mode][2] / 'bin/python').exists() or (mode == 'lstm' and not (ROOT / '.venv/bin/python').exists()):
        return jsonify(error='This analysis environment is not installed. See the README setup instructions.'), 503
    uploads = {'input': request.files.get('video')}
    if mode == 'lstm':
        uploads['reference'] = request.files.get('reference')
    for role, video in uploads.items():
        if not video or Path(video.filename or '').suffix.lower() not in {'.mp4', '.mov', '.avi'}:
            label = 'reference video' if role == 'reference' else 'movement video'
            return jsonify(error=f'Choose an MP4, MOV, or AVI {label}.'), 400
    if not slots.acquire(blocking=False):
        return jsonify(error='The analysis queue is full. Please try again after a job finishes.'), 429
    path = Path(app.config['JOB_ROOT']) / uuid.uuid4().hex
    try:
        path.mkdir(parents=True)
        for role, video in uploads.items():
            target = path / (role + Path(video.filename).suffix.lower())
            video.save(target)
            if not 0 < target.stat().st_size <= 250 * 1024 * 1024:
                shutil.rmtree(path)
                slots.release()
                return jsonify(error=f'The {role} video must be nonempty and no larger than 250 MB.'), 400
        filename = Path(uploads['input'].filename).name
        reference_filename = Path(uploads['reference'].filename).name if mode == 'lstm' else None
        save_status(path, state='queued', filename=filename, reference_filename=reference_filename, mode=mode)
        executor.submit(analyze, path, mode, filename, reference_filename)
    except Exception:
        slots.release()
        raise
    return jsonify(id=path.name), 202


@app.get('/api/jobs/<job_id>')
def status(job_id):
    path = job_dir(job_id)
    data = json.loads((path / 'status.json').read_text())
    if data['state'] == 'running' and not data.get('stage') and (path / 'analysis.log').exists():
        matches = re.findall(r'Processed (\d+)/(\d+) frames', (path / 'analysis.log').read_text(errors='replace'))
        if matches:
            done, total = map(int, matches[-1])
            data['progress'] = min(99, round(done / max(1, total) * 100))
    if data['state'] == 'complete':
        data['files'] = [{'name': p.name, 'url': f'/api/jobs/{job_id}/files/{p.name}'}
                         for p in sorted((path / 'output').iterdir()) if p.is_file()]
    data['log_url'] = f'/api/jobs/{job_id}/log'
    return jsonify(data)


@app.get('/api/jobs/<job_id>/files/<filename>')
def artifact(job_id, filename):
    path = job_dir(job_id)
    if json.loads((path / 'status.json').read_text())['state'] != 'complete':
        abort(404)
    return send_from_directory(path / 'output', filename, as_attachment=request.args.get('download') == '1')


@app.get('/api/jobs/<job_id>/log')
def log(job_id):
    return send_from_directory(job_dir(job_id), 'analysis.log', mimetype='text/plain')


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(error='Upload is too large. Each video must be at most 250 MB (500 MB combined).'), 413


if __name__ == '__main__':
    # Jobs belong to a single local process; flag interrupted work on restart.
    for status_file in Path(app.config['JOB_ROOT']).glob('*/status.json'):
        state = json.loads(status_file.read_text())
        if state['state'] in {'queued', 'running'}:
            state.update(state='error', error='The server restarted. Please upload your video again.')
            save_status(status_file.parent, **state)
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5050)), threaded=True, debug=False)
