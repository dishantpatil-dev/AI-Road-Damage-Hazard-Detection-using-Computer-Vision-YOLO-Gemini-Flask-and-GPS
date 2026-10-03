import base64
import sys
from pathlib import Path

import pytest

# app.py imports detectors/storage/verifier as top-level modules.
# Add the backend directory to Python's import path for API tests.
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    return app.app.test_client()


def test_analyze_rejects_invalid_base64(client):
    response = client.post(
        "/api/analyze",
        json={"image": "this-is-not-valid-base64"},
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "invalid image payload"
    }


def test_analyze_accepts_valid_base64(client, monkeypatch):
    image_bytes = b"fake-image-data"
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    captured = {}

    def fake_analyze_frame(image, lat, lon):
        captured["image"] = image
        captured["lat"] = lat
        captured["lon"] = lon

        return [
            {
                "detector": "pothole",
                "label": "Pothole",
                "status": "clear",
                "message": "No candidate above threshold",
            }
        ]

    monkeypatch.setattr(app, "analyze_frame", fake_analyze_frame)

    response = client.post(
        "/api/analyze",
        json={
            "image": f"data:image/jpeg;base64,{encoded}",
            "lat": 18.76,
            "lon": 73.85,
        },
    )

    assert response.status_code == 200

    body = response.get_json()

    assert body["results"][0]["status"] == "clear"
    assert captured["image"] == image_bytes
    assert captured["lat"] == 18.76
    assert captured["lon"] == 73.85


def test_analyze_video_requires_video_file(client):
    response = client.post(
        "/api/analyze-video",
        data={
            "lat": "18.76",
            "lon": "73.85",
        },
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "no video file provided"
    }


def test_pins_returns_storage_results(client, monkeypatch):
    expected_pins = [
        {
            "id": "pin-1",
            "type": "pothole",
            "label": "Pothole",
            "lat": 18.76,
            "lon": 73.85,
            "confidence": 0.91,
            "reason": "Test pin",
            "image": None,
        }
    ]

    monkeypatch.setattr(
        app.storage,
        "get_pins",
        lambda: expected_pins,
    )

    response = client.get("/api/pins")

    assert response.status_code == 200
    assert response.get_json() == {
        "pins": expected_pins
    }


def test_status_uses_heuristic_without_gemini_key(client, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("HEURISTIC_CONFIRM_THRESHOLD", "0.65")

    response = client.get("/api/status")

    assert response.status_code == 200

    body = response.get_json()

    assert body["verification_mode"] == "heuristic"
    assert body["heuristic_threshold"] == 0.65


def test_analyze_rejects_missing_image(client):
    response = client.post(
        "/api/analyze",
        json={},
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "invalid image payload"
    }


def test_analyze_rejects_malformed_json(client):
    response = client.post(
        "/api/analyze",
        data='{"image": ',
        content_type="application/json",
    )

    assert response.status_code == 400


def test_analyze_rejects_non_object_json(client):
    response = client.post(
        "/api/analyze",
        json=[],
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "invalid image payload"
    }
