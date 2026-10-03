import base64
import os
import time
import uuid

import cv2
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

import detectors
import storage
import verifier

load_dotenv()

APP_ROOT = os.path.dirname(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(APP_ROOT, "frontend")
UPLOADS_DIR = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = float(os.environ.get("DETECTION_CONFIDENCE_THRESHOLD", 0.4))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


def _save_frame(image_bytes: bytes) -> str:
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOADS_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return f"/uploads/{filename}"


def analyze_frame(image_bytes: bytes, lat, lon):
    """Runs the full pipeline on one frame:
    Roboflow (x3) -> confidence gate -> Gemini confirm -> pin if confirmed.
    Returns a list of per-detector outcome dicts for the UI log.
    """
    outcomes = []
    detector_results = detectors.run_all_detectors(image_bytes)

    for det in detector_results:
        if det["error"]:
            outcomes.append({
                "detector": det["detector"], "label": det["label"],
                "status": "error", "message": det["error"],
            })
            continue

        top = detectors.best_prediction(det["predictions"])
        if not top or top.get("confidence", 0) < CONFIDENCE_THRESHOLD:
            outcomes.append({
                "detector": det["detector"], "label": det["label"],
                "status": "clear", "message": "No candidate above threshold",
            })
            continue

        # Candidate found -> confirm before it becomes a pin (Gemini if
        # configured, otherwise the confidence-heuristic fallback)
        verdict = verifier.verify_detection(
            image_bytes, det["label"], roboflow_confidence=top.get("confidence", 0)
        )

        if verdict["error"]:
            outcomes.append({
                "detector": det["detector"], "label": det["label"],
                "status": "error", "message": f"Verification error: {verdict['error']}",
                "roboflow_confidence": top.get("confidence"),
            })
            continue

        if verdict["confirmed"] and lat is not None and lon is not None:
            image_path = _save_frame(image_bytes)
            pin = storage.add_pin(
                detector=det["detector"], label=det["label"],
                lat=lat, lon=lon,
                confidence=verdict["confidence"] or top.get("confidence", 0),
                reason=verdict["reason"], image_path=image_path,
            )
            outcomes.append({
                "detector": det["detector"], "label": det["label"],
                "status": "confirmed",
                "message": f"[{verdict['method']}] {verdict['reason']}", "pin": pin,
            })
        elif verdict["confirmed"] and (lat is None or lon is None):
            outcomes.append({
                "detector": det["detector"], "label": det["label"],
                "status": "error",
                "message": "Confirmed but no location available yet — waiting for GPS",
            })
        else:
            outcomes.append({
                "detector": det["detector"], "label": det["label"],
                "status": "rejected",
                "message": f"[{verdict['method']}] " + (verdict["reason"] or "Not confirmed"),
                "roboflow_confidence": top.get("confidence"),
            })

    return outcomes


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Body: {"image": "data:image/jpeg;base64,...", "lat": float, "lon": float}
    Used by the live webcam view — one frame at a time."""
    payload = request.get_json(force=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "invalid image payload"}), 400

    data_url = payload.get("image", "")
    lat = payload.get("lat")
    lon = payload.get("lon")

    if not data_url:
        return jsonify({"error": "invalid image payload"}), 400

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(data_url, validate=True)
    except Exception:
        return jsonify({"error": "invalid image payload"}), 400

    outcomes = analyze_frame(image_bytes, lat, lon)
    return jsonify({"results": outcomes})


@app.route("/api/analyze-video", methods=["POST"])
def api_analyze_video():
    """Multipart form: video file + lat + lon (single fixed location, since
    an uploaded file has no live GPS track). Samples 1 frame every N
    seconds of video and runs the same pipeline on each."""
    if "video" not in request.files:
        return jsonify({"error": "no video file provided"}), 400

    lat = request.form.get("lat", type=float)
    lon = request.form.get("lon", type=float)
    sample_every_sec = request.form.get("sample_every_sec", default=2, type=float)

    video_file = request.files["video"]
    temp_path = os.path.join(UPLOADS_DIR, f"upload_{uuid.uuid4().hex}.mp4")
    video_file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps * sample_every_sec))

    frame_idx = 0
    all_outcomes = []
    frames_processed = 0
    MAX_FRAMES = 40  # safety cap so one upload can't run forever

    while cap.isOpened() and frames_processed < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                outcomes = analyze_frame(buf.tobytes(), lat, lon)
                all_outcomes.append({
                    "frame_time_sec": round(frame_idx / fps, 1),
                    "results": outcomes,
                })
                frames_processed += 1
        frame_idx += 1

    cap.release()
    os.remove(temp_path)

    return jsonify({"frames_analyzed": frames_processed, "frames": all_outcomes})


@app.route("/api/pins", methods=["GET"])
def api_pins():
    return jsonify({"pins": storage.get_pins()})


@app.route("/api/status", methods=["GET"])
def api_status():
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    has_gemini = bool(gemini_key) and "your_" not in gemini_key
    return jsonify({
        "verification_mode": "gemini" if has_gemini else "heuristic",
        "heuristic_threshold": float(os.environ.get("HEURISTIC_CONFIRM_THRESHOLD", 0.65)),
        "detection_threshold": CONFIDENCE_THRESHOLD,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
