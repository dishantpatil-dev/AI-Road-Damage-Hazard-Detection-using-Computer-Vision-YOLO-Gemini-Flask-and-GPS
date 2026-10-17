# 🚦 CivicScan — AI Road Damage & Hazard Detection

<p align="center">
  <strong>Detect road issues. Verify candidates with AI. Map confirmed reports.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS?style=for-the-badge&logo=github&label=STARS" alt="GitHub stars" />
  <img src="https://img.shields.io/github/forks/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS?style=for-the-badge&logo=github&label=FORKS" alt="GitHub forks" />
  <img src="https://img.shields.io/github/last-commit/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS?style=for-the-badge&logo=github&label=LAST%20COMMIT" alt="Last commit" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+" />
</p>

<p align="center">
  <a href="https://civicscan-road-intelligence.onrender.com">🌐 Live Demo</a>
  ·
  <a href="https://github.com/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS">📦 Source Code</a>
  ·
  <a href="https://github.com/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS/issues">🐛 Report a Bug</a>
  ·
  <a href="https://github.com/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS/issues/new">💡 Request a Feature</a>
</p>

---

## 📌 Overview

**CivicScan** is a computer-vision civic-tech prototype that analyzes camera frames and video for potential road issues, applies an additional AI verification step, and displays confirmed detections on a map.

The workflow is designed to separate initial detection from verification:

**Capture → Detect → Verify → Geolocate → Map**

The goal is to make road-issue observations easier to review and visualize. CivicScan is a prototype and should not be treated as an authoritative safety or emergency-reporting system.

## ✨ Key Features

| Feature | What it does |
|---|---|
| 🎥 Live camera scanning | Captures frames from a browser camera for analysis |
| 📹 Video analysis | Processes uploaded video footage |
| 🤖 Road-issue detection | Uses configured Roboflow-hosted computer-vision models to identify potential issues such as potholes, garbage areas, and road encroachment |
| 🧠 AI verification | Sends eligible candidates to Gemini for an additional verification step |
| 📍 GPS coordinates | Uses browser geolocation where available to associate observations with a location |
| 🗺️ Interactive map | Displays confirmed issue pins with Leaflet and OpenStreetMap |
| 📋 Detection activity log | Presents detection and verification outcomes, including clear, rejected, confirmed, or error events |
| ⚙️ Tunable scanning | Configures the detection confidence threshold and capture interval |
| 🔌 Flask REST API | Provides backend endpoints for frame/video analysis and map-pin data |
| 🧪 Automated tests and CI | Runs the test suite through GitHub Actions on pushes and pull requests to `main` |

## 🖥️ Demo

<p align="center">
  <img width="100%" src="https://github.com/user-attachments/assets/95deb616-a328-44ad-8b5f-a4a56fd651e9" alt="CivicScan road issue detection interface" />
</p>

Try the [CivicScan live demo](https://civicscan-road-intelligence.onrender.com).

> Camera and geolocation features require browser permission. On deployed sites, these features generally require HTTPS.

## 🧠 How It Works

1. **Capture:** The user starts a browser-camera scan or supplies a video.
2. **Detect:** Configured Roboflow models identify potential road-issue candidates.
3. **Filter:** Candidates are compared with the configured confidence threshold.
4. **Verify:** Eligible candidates are sent to Gemini for additional verification.
5. **Locate:** Available GPS coordinates are associated with the observation.
6. **Visualize:** Confirmed issues are stored by the prototype and shown as pins on the interactive map.
7. **Review:** The activity log records the outcome of the processing workflow.

The verification stage is intended to reduce unverified candidates becoming confirmed map entries; it does not guarantee that every detection is correct.

## 🧰 Technology Stack

- **Language:** Python
- **Backend:** Flask, REST API
- **Computer vision:** OpenCV, Roboflow-hosted detection models
- **AI verification:** Google Gemini API
- **Frontend:** HTML, CSS, JavaScript
- **Mapping:** Leaflet, OpenStreetMap, browser geolocation
- **Testing and automation:** pytest, GitHub Actions
- **Version control:** Git, GitHub

## 🏗️ Project Structure

The repository's main application components are organized around the Flask backend, browser interface, detection/verification pipeline, and prototype map-pin storage. Check the source tree for the current file layout before relying on individual paths.

## 🚀 Run Locally

### Prerequisites

- Python 3.9 or later
- Git
- A webcam or a sample video for scanning
- A Google Gemini API key
- Roboflow API credentials and model identifiers for the configured detection models

### 1. Clone the repository

```bash
git clone https://github.com/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS.git
cd AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS
```

### 2. Create and activate a virtual environment

**Windows (PowerShell or Command Prompt):**

```bash
python -m venv venv
venv\\Scripts\\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure API credentials

Create a local `.env` file using the variable names expected by the application. For the current model configuration, the environment file uses a Gemini key and credentials/model IDs for the pothole, garbage, and encroachment detectors.

Example template:

```ini
GOOGLE_API_KEY=your_google_api_key

ROBOFLOW_POTHOLE_API_KEY=your_api_key
ROBOFLOW_POTHOLE_MODEL=your-project/1

ROBOFLOW_GARBAGE_API_KEY=your_api_key
ROBOFLOW_GARBAGE_MODEL=your-project/1

ROBOFLOW_ENCROACHMENT_API_KEY=your_api_key
ROBOFLOW_ENCROACHMENT_MODEL=your-project/1
```

Replace the example values with your own credentials and model IDs. **Never commit real API keys or your `.env` file to GitHub.**

### 5. Start the application

```bash
cd backend
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## ⚙️ Configuration

| Setting | Purpose | Default documented by the project |
|---|---|---|
| `DETECTION_CONFIDENCE_THRESHOLD` | Minimum detector confidence required before Gemini verification | `0.4` |
| `CAPTURE_INTERVAL_SECONDS` | Delay between captured frames | `4` |

Lower thresholds may increase candidate detections and verification requests. Higher thresholds may reduce requests but can miss potential issues.

## 🔌 API Endpoints

The backend includes endpoints such as:

| Endpoint | Purpose |
|---|---|
| `/api/analyze` | Analyze a single frame |
| `/api/analyze-video` | Analyze frames from an uploaded video |
| `/api/pins` | Retrieve confirmed map pins |

For exact request and response formats, refer to the current backend implementation.

## 🧪 Tests

Install the development requirements if the repository's development dependency file is available:

```bash
pip install -r requirements-dev.txt
```

Run the test suite from the repository root:

```bash
pytest -q
```

GitHub Actions is configured to run automated tests for pushes and pull requests to `main`.

## ⚠️ Limitations

CivicScan is an experimental prototype, not a production-grade traffic or emergency monitoring service.

- Detection quality depends on model performance, input quality, lighting, and the selected confidence threshold.
- Hosted inference services may impose rate limits or incur usage costs.
- Gemini verification may add latency and API cost.
- Browser geolocation requires permission and may be unavailable; uploaded video does not inherently provide a GPS track.
- The prototype uses simple JSON-based storage for confirmed pins, which is not designed for concurrent, large-scale production workloads.
- Do not use detections as the sole basis for urgent safety decisions.

## 🗺️ Roadmap

- [ ] Move confirmed-pin storage to a database such as PostgreSQL/PostGIS
- [ ] Add authentication and user accounts
- [ ] Build historical incident analytics
- [ ] Improve issue filtering and severity classification
- [ ] Add performance benchmarks across representative hardware and video inputs
- [ ] Publish detailed API request/response documentation
- [ ] Add a short demo video or GIF
- [x] Automated tests
- [x] GitHub Actions CI

## 🤝 Contributing

Contributions, bug reports, and feature ideas are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Run the tests locally.
5. Open a pull request describing the change and test results.

Use the [Issues page](https://github.com/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS/issues) to report bugs or suggest improvements.

## ⭐ Support CivicScan

If CivicScan is useful for learning computer vision, experimenting with AI-assisted verification, or exploring civic-tech ideas, consider starring the repository.

<p align="center">
  <a href="https://github.com/dishantpatil-dev/AI-Road-Damage-Hazard-Detection-using-Computer-Vision-YOLO-Gemini-Flask-and-GPS">
    <img src="https://img.shields.io/badge/⭐_Star_CivicScan-on_GitHub-181717?style=for-the-badge&logo=github" alt="Star CivicScan on GitHub" />
  </a>
</p>

---

<p align="center">
  <strong>Built with Python · Computer Vision · AI · Maps</strong>
</p>
