// ---------------------------------------------------------------
// CivicScan frontend — talks to the Flask backend at the same origin
// ---------------------------------------------------------------

const TYPE_COLORS = {
  pothole: '#ff7a45',
  garbage: '#53c7ff',
  encroachment: '#51e0a1',
};

let currentLat = null;
let currentLon = null;
let scanning = false;
let scanTimer = null;
let scanInFlight = false;
let mediaStream = null;
let map, pinLayer;

// ---------- Tabs ----------
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.remove('hidden');
  });
});

// ---------- Geolocation ----------
function initGeolocation() {
  const locDot = document.getElementById('locDot');
  const locText = document.getElementById('locText');

  if (!navigator.geolocation) {
    locDot.classList.add('error');
    locText.textContent = 'Geolocation not supported — enter coordinates manually in Upload tab';
    return;
  }

  navigator.geolocation.watchPosition(
    (pos) => {
      currentLat = pos.coords.latitude;
      currentLon = pos.coords.longitude;
      locDot.classList.add('ok');
      locDot.classList.remove('error');
      locText.textContent = `${currentLat.toFixed(5)}, ${currentLon.toFixed(5)}`;
    },
    (err) => {
      locDot.classList.add('error');
      locText.textContent = `Location unavailable (${err.message}) — enter manually in Upload tab`;
    },
    { enableHighAccuracy: true, maximumAge: 5000 }
  );
}

// ---------- Backend health check ----------
async function checkBackend() {
  const dot = document.getElementById('apiStatusDot');
  const text = document.getElementById('apiStatusText');
  try {
    const [pinsRes, statusRes] = await Promise.all([
      fetch('/api/pins'),
      fetch('/api/status'),
    ]);
    if (!pinsRes.ok) throw new Error('bad response');
    const status = await statusRes.json();
    dot.className = 'dot dot-ok';
    text.textContent = status.verification_mode === 'gemini'
      ? 'pipeline reachable · Gemini verification'
      : `pipeline reachable · heuristic verification (≥${Math.round(status.heuristic_threshold * 100)}%, no Gemini key)`;
  } catch (e) {
    dot.className = 'dot dot-error';
    text.textContent = 'backend unreachable — is app.py running?';
  }
}

// ---------- Webcam ----------
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const scanOverlay = document.getElementById('scanOverlay');

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = mediaStream;
  } catch (e) {
    logEntry({ status: 'error', label: 'Camera', message: 'Could not access camera: ' + e.message });
    return false;
  }
  return true;
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
}

function captureFrameDataUrl() {
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.85);
}

async function scanOnce() {
  // setInterval can fire again while a slow analysis is still running.
  // Skip overlapping captures so one camera session cannot flood the backend.
  if (!scanning || scanInFlight) return;
  scanInFlight = true;
  try {
    const dataUrl = captureFrameDataUrl();
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl, lat: currentLat, lon: currentLon }),
    });
    const data = await res.json();
    (data.results || []).forEach(logEntry);
    (data.results || []).forEach(r => { if (r.pin) addPinToMap(r.pin); });
  } catch (e) {
    logEntry({ status: 'error', label: 'Network', message: e.message });
  } finally {
    scanInFlight = false;
  }
}

startBtn.addEventListener('click', async () => {
  const ok = await startCamera();
  if (!ok) return;
  scanning = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  scanOverlay.classList.add('scanning');
  const intervalSec = Math.max(2, parseInt(document.getElementById('intervalInput').value, 10) || 4);
  scanTimer = setInterval(scanOnce, intervalSec * 1000);
  scanOnce(); // immediate first scan
});

stopBtn.addEventListener('click', () => {
  scanning = false;
  clearInterval(scanTimer);
  stopCamera();
  startBtn.disabled = false;
  stopBtn.disabled = true;
  scanOverlay.classList.remove('scanning');
});

// ---------- Video upload ----------
const uploadDrop = document.getElementById('uploadDrop');
const videoFileInput = document.getElementById('videoFile');
const browseLink = document.getElementById('browseLink');
const fileNameEl = document.getElementById('fileName');
let selectedVideoFile = null;

browseLink.addEventListener('click', () => videoFileInput.click());
uploadDrop.addEventListener('click', (e) => { if (e.target === uploadDrop) videoFileInput.click(); });

['dragover', 'dragenter'].forEach(evt =>
  uploadDrop.addEventListener(evt, (e) => { e.preventDefault(); uploadDrop.classList.add('drag-over'); })
);
['dragleave', 'drop'].forEach(evt =>
  uploadDrop.addEventListener(evt, (e) => { e.preventDefault(); uploadDrop.classList.remove('drag-over'); })
);
uploadDrop.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) setSelectedFile(e.dataTransfer.files[0]);
});
videoFileInput.addEventListener('change', () => {
  if (videoFileInput.files.length) setSelectedFile(videoFileInput.files[0]);
});
function setSelectedFile(file) {
  selectedVideoFile = file;
  fileNameEl.textContent = `${file.name} (${(file.size / 1e6).toFixed(1)} MB)`;
}

document.getElementById('useMyLocBtn').addEventListener('click', () => {
  if (currentLat !== null) {
    document.getElementById('manualLat').value = currentLat.toFixed(6);
    document.getElementById('manualLon').value = currentLon.toFixed(6);
  }
});

document.getElementById('analyzeVideoBtn').addEventListener('click', async () => {
  const progress = document.getElementById('videoProgress');
  if (!selectedVideoFile) { progress.textContent = 'Choose a video file first.'; return; }

  const lat = parseFloat(document.getElementById('manualLat').value);
  const lon = parseFloat(document.getElementById('manualLon').value);
  if (Number.isNaN(lat) || Number.isNaN(lon)) {
    progress.textContent = 'Enter a latitude and longitude (or click "Use my location").';
    return;
  }

  const sampleEvery = document.getElementById('videoIntervalInput').value || 2;
  const formData = new FormData();
  formData.append('video', selectedVideoFile);
  formData.append('lat', lat);
  formData.append('lon', lon);
  formData.append('sample_every_sec', sampleEvery);

  progress.textContent = 'Uploading and analyzing — this can take a minute…';
  try {
    const res = await fetch('/api/analyze-video', { method: 'POST', body: formData });
    const data = await res.json();
    progress.textContent = `Analyzed ${data.frames_analyzed} sampled frame(s).`;
    (data.frames || []).forEach(frame => {
      frame.results.forEach(r => {
        logEntry(r);
        if (r.pin) addPinToMap(r.pin);
      });
    });
  } catch (e) {
    progress.textContent = 'Error analyzing video: ' + e.message;
  }
});

// ---------- Log ----------
const logList = document.getElementById('logList');
function logEntry(r) {
  const empty = logList.querySelector('.log-empty');
  if (empty) empty.remove();

  const el = document.createElement('div');
  el.className = `log-entry ${r.status}`;
  const time = new Date().toLocaleTimeString();
  el.innerHTML = `
    <div style="flex:1">
      <span class="lbl">${r.label || r.detector || 'Detector'}</span>
      <span class="tag tag-${r.status}">${r.status}</span>
      <span class="msg">${r.message || ''} · ${time}</span>
    </div>`;
  logList.prepend(el);
  while (logList.children.length > 60) logList.removeChild(logList.lastChild);
}

// ---------- Map ----------
function initMap() {
  map = L.map('map', { zoomControl: true }).setView([20.5937, 78.9629], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(map);
  pinLayer = L.layerGroup().addTo(map);
}

const TYPE_LABELS = {
  pothole: 'P',
  garbage: 'G',
  encroachment: 'E',
};

function addPinToMap(pin, { fly = true } = {}) {
  const color = TYPE_COLORS[pin.type] || '#FFFFFF';
  const glyph = TYPE_LABELS[pin.type] || '?';
  const icon = L.divIcon({
    className: 'pin-icon',
    html: `<div class="pin-glyph" style="background:${color}">${glyph}</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
  const marker = L.marker([pin.lat, pin.lon], { icon });
  const when = new Date(pin.timestamp * 1000).toLocaleString();
  const imgTag = pin.image
    ? `<img class="popup-thumb" src="${pin.image}" alt="${pin.label}" />`
    : '';
  marker.bindPopup(`
    ${imgTag}
    <div class="popup-title">${pin.label}</div>
    <div class="popup-meta">confidence ${(pin.confidence * 100).toFixed(0)}%</div>
    <div class="popup-meta">${when}</div>
    <div class="popup-meta">${pin.reason || ''}</div>
  `);
  marker.addTo(pinLayer);
  if (fly) map.flyTo([pin.lat, pin.lon], Math.max(map.getZoom(), 15), { duration: 0.8 });
  updatePinCount();
}

function updatePinCount() {
  document.getElementById('pinCount').textContent = `${pinLayer.getLayers().length} pinned`;
}

async function loadExistingPins() {
  try {
    const res = await fetch('/api/pins');
    const data = await res.json();
    (data.pins || []).forEach(p => addPinToMap(p, { fly: false }));
    if (data.pins && data.pins.length) {
      const last = data.pins[data.pins.length - 1];
      map.setView([last.lat, last.lon], 13);
    }
  } catch (e) { /* backend not up yet — fine, health check will report it */ }
}

// ---------- Boot ----------
initGeolocation();
checkBackend();
initMap();
loadExistingPins();
