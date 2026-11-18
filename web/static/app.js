const $ = id => document.getElementById(id);
const descriptions = {
  lstm: 'Upload both videos. A new LSTM learns from your reference and measures deviation in your movement. This includes training, so comparison may take several minutes.',
  body: 'Track 33 body landmarks and major joint angles. Keep your whole body visible and the camera steady.',
  wrist: 'Track wrist angles on both sides. Film from the side for flexion and extension, with your hand and elbow visible.',
  yolo: 'Detect body pose and measure shoulder, elbow, hip, and knee angles using YOLO.'
};
let selectedFile = null;
let referenceFile = null;
let busy = false;
let activeJob = null;
let pollTimer;
function error(message) { $('form-error').textContent = message; $('form-error').hidden = !message; }
function selectFile(file) {
  if (busy || !file) return;
  error('');
  selectedFile = null;
  $('file-title').textContent = 'Drop your video here';
  $('file-detail').textContent = 'or browse files';
  if (!/\.(mp4|mov|avi)$/i.test(file.name)) { error('Please choose an MP4, MOV, or AVI video.'); return; }
  if (!file.size || file.size > 250 * 1024 * 1024) { error('Choose a nonempty video up to 250 MB.'); return; }
  selectedFile = file;
  $('file-title').textContent = file.name;
  $('file-detail').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · Click to change`;
}
$('video-file').addEventListener('change', e => selectFile(e.target.files[0]));
function updateMode() {
  $('mode-help').textContent = descriptions[$('mode').value];
  $('reference-upload').hidden = $('mode').value !== 'lstm';
  $('reference-file').required = $('mode').value === 'lstm' && !referenceFile;
}
$('mode').addEventListener('change', updateMode);
function selectReference(file) {
  if (busy || !file) return;
  error(''); referenceFile = null;
  $('reference-title').textContent = 'Drop your reference video here';
  $('reference-detail').textContent = 'or browse files';
  if (!/\.(mp4|mov|avi)$/i.test(file.name) || !file.size || file.size > 250 * 1024 * 1024) {
    error('Choose a nonempty MP4, MOV, or AVI reference video up to 250 MB.'); updateMode(); return;
  }
  referenceFile = file;
  $('reference-title').textContent = file.name;
  $('reference-detail').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · Click to change`;
  updateMode();
}
$('reference-file').addEventListener('change', e => selectReference(e.target.files[0]));
['dragenter','dragover'].forEach(type => $('reference-dropzone').addEventListener(type, e => { e.preventDefault(); if (!busy) $('reference-dropzone').classList.add('dragging'); }));
['dragleave','drop'].forEach(type => $('reference-dropzone').addEventListener(type, e => { e.preventDefault(); $('reference-dropzone').classList.remove('dragging'); }));
$('reference-dropzone').addEventListener('drop', e => selectReference(e.dataTransfer.files[0]));
['dragenter','dragover'].forEach(type => $('dropzone').addEventListener(type, e => {e.preventDefault(); if (!busy) $('dropzone').classList.add('dragging');}));
['dragleave','drop'].forEach(type => $('dropzone').addEventListener(type, e => {e.preventDefault(); $('dropzone').classList.remove('dragging');}));
$('dropzone').addEventListener('drop', e => selectFile(e.dataTransfer.files[0]));
function setBusy(value) { busy = value; ['submit','mode','video-file','reference-file'].forEach(id => $(id).disabled = value); $('submit').textContent = value ? 'Analysis in progress…' : 'Analyze video ↗'; }
function panel(id) { ['empty-state','progress-panel','result-content','failure'].forEach(name => $(name).hidden = name !== id); }
function fail(message, logUrl) {
  clearTimeout(pollTimer); setBusy(false); panel('failure'); $('status-badge').textContent = 'NEEDS ATTENTION';
  $('failure-message').textContent = message; $('log-link').hidden = !logUrl;
  if (logUrl) $('log-link').href = logUrl;
  sessionStorage.removeItem('conradJob'); activeJob = null;
}
function upload(formData) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open('POST', '/api/jobs'); xhr.setRequestHeader('X-Conrad-Upload', '1');
    xhr.upload.onprogress = e => { if (e.lengthComputable) { $('progress').value = Math.round(100 * e.loaded / e.total); $('progress-detail').textContent = `Uploading · ${$('progress').value}%`; } };
    xhr.onload = () => { let data; try { data = JSON.parse(xhr.responseText); } catch { reject(new Error('The server returned an unexpected response. Please try again.')); return; } if (xhr.status === 202) resolve(data); else reject(new Error(data.error || 'Upload failed. Please try again.')); };
    xhr.onerror = () => reject(new Error('Could not reach the analysis server. Check that it is running.'));
    xhr.send(formData);
  });
}
$('upload-form').addEventListener('submit', async e => {
  e.preventDefault(); if (busy) return; error('');
  if (!selectedFile) { error('Choose a video first.'); $('video-file').focus(); return; }
  if ($('mode').value === 'lstm' && !referenceFile) { error('Choose a reference video too.'); $('reference-file').focus(); return; }
  const data = new FormData(); data.append('video', selectedFile); data.append('mode', $('mode').value);
  if ($('mode').value === 'lstm') data.append('reference', referenceFile);
  setBusy(true); panel('progress-panel'); $('status-badge').textContent = 'UPLOADING';
  $('progress-title').textContent = 'Uploading your video files…'; $('progress-detail').textContent = 'Preparing the upload.'; $('progress').value = 0;
  try { const job = await upload(data); activeJob = job.id; sessionStorage.setItem('conradJob', job.id); poll(job.id); }
  catch (err) { fail(err.message); }
});
async function poll(id) {
  try {
    const response = await fetch(`/api/jobs/${id}`);
    if (!response.ok) throw new Error('This analysis is unavailable. Please upload your video again.');
    const data = await response.json();
    if (data.state === 'complete') { renderResult(data); sessionStorage.removeItem('conradJob'); activeJob = null; setBusy(false); return; }
    if (data.state === 'error') { fail(data.error, data.log_url); return; }
    $('status-badge').textContent = data.state === 'queued' ? 'IN QUEUE' : 'ANALYZING';
    $('progress-title').textContent = data.stage || (data.state === 'queued' ? 'Your video is in the queue.' : 'Following your movement…');
    $('progress-detail').textContent = data.stage ? 'Processing your uploaded pair. Reference extraction, model training, and comparison run in sequence.' : data.progress != null ? `${data.progress}% of frames processed. Charts and video preview are prepared next.` : 'Processing your video. The first update may take a moment.';
    if (data.progress != null && !data.stage) $('progress').value = data.progress; else $('progress').removeAttribute('value');
    pollTimer = setTimeout(() => poll(id), 1800);
  } catch (err) { fail(err.message); }
}
function metric(value, label) { const div = document.createElement('div'); div.className = 'metric'; const strong = document.createElement('strong'); strong.textContent = value; const span = document.createElement('span'); span.textContent = label; div.append(strong, span); return div; }
const fmt = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '—';
function renderResult(data) {
  panel('result-content'); $('status-badge').textContent = 'ANALYSIS COMPLETE'; $('result-name').textContent = data.filename;
  $('result-reference').hidden = !data.reference_filename;
  $('result-reference').textContent = data.reference_filename ? `Reference: ${data.reference_filename} · Preview shows both videos and the deviation timeline.` : '';
  const summary = data.summary; const files = data.files;
  const warning = [data.warning, summary.lstm_warning].filter(Boolean).join(' ');
  $('result-warning').hidden = !warning; $('result-warning').textContent = warning;
  const video = files.find(f => f.name === 'preview.mp4'); $('result-video').hidden = !video;
  if (video) $('result-video').src = video.url; else { $('result-video').removeAttribute('src'); $('result-video').load(); }
  $('metrics').replaceChildren(metric(summary.frames_processed, 'frames analyzed'), metric(`${fmt(summary.frames_processed / summary.fps)}s`, 'video duration'));
  const coverage = summary.pose_coverage_percent ?? summary.person_coverage_percent;
  if (coverage != null) $('metrics').append(metric(`${fmt(coverage)}%`, 'pose coverage'));
  $('lstm-result').hidden = !summary.lstm;
  if (summary.lstm) {
    const strong = document.createElement('strong'); strong.textContent = `${fmt(summary.lstm.percent_above_threshold)}%`;
    const title = document.createElement('span'); title.textContent = 'of windows above the reference deviation threshold';
    const note = document.createElement('p'); note.textContent = `${summary.lstm.windows} windows compared. Mean reconstruction error: ${fmt(summary.lstm.mean_error)}. Higher deviation means less similarity to your uploaded reference, not an unsafe or incorrect movement.`;
    $('lstm-result').replaceChildren(strong, title, note);
  }
  const deviation = files.find(f => f.name === 'lstm_deviation_timeline.png');
  $('deviation-chart').hidden = !deviation;
  if (deviation) $('deviation-chart').src = deviation.url;
  const chart = files.find(f => f.name.startsWith('input_') && f.name.endsWith('.png')); $('result-chart').hidden = !chart;
  if (chart) $('result-chart').src = chart.url;
  $('joint-rows').replaceChildren();
  Object.entries(summary.angles || summary.sides || {}).forEach(([name, values]) => {
    const row = document.createElement('tr');
    [name.replaceAll('_', ' '), `${fmt(values.minimum_deg)}°`, `${fmt(values.maximum_deg)}°`, `${fmt(values.range_deg ?? values.range_of_motion_deg)}°`, `${fmt(values.coverage_percent)}%`].forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); });
    $('joint-rows').append(row);
  });
  $('downloads').replaceChildren();
  files.forEach(file => { const link = document.createElement('a'); link.href = `${file.url}?download=1`; link.textContent = `↓ ${file.name.replace(/^input_/, '')}`; $('downloads').append(link); });
}
$('retry').addEventListener('click', () => { panel('empty-state'); $('status-badge').textContent = 'READY WHEN YOU ARE'; $('submit').focus(); });
async function initialize() {
  try {
    const response = await fetch('/api/modes'); const modes = await response.json();
    modes.forEach(mode => { const option = [...$('mode').options].find(o => o.value === mode.id); if (option) { option.disabled = !mode.available; if (!mode.available) option.textContent += ' (not installed)'; } });
    if ($('mode').selectedOptions[0]?.disabled) { const available = [...$('mode').options].find(o => !o.disabled); if (available) $('mode').value = available.value; }
    updateMode();
  } catch { error('Cannot connect to the analysis server. Start it and reload this page.'); }
  try {
    const response = await fetch('/api/example'); const data = await response.json();
    if (data.available) { $('example-video').src = data.video; $('example-video').poster = data.poster; $('example-windows').textContent = data.windows; $('example-features').textContent = data.features; }
    else { $('example-video').hidden = true; $('example-fallback').hidden = false; }
  } catch { $('example-video').hidden = true; $('example-fallback').hidden = false; }
  const previous = sessionStorage.getItem('conradJob');
  if (previous && /^[a-f0-9]{32}$/.test(previous)) { activeJob = previous; setBusy(true); panel('progress-panel'); poll(previous); }
}
initialize();
