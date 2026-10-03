import json

from unittest.mock import Mock
from backend import detectors
from backend import storage
from backend import verifier


def test_find_predictions_from_nested_workflow_response():
    payload = {
        "output": {
            "predictions": [
                {"confidence": 0.31, "x": 10},
                {"confidence": 0.87, "x": 20},
            ]
        }
    }

    predictions = detectors._find_predictions(payload)

    assert predictions is not None
    assert len(predictions) == 2
    assert detectors.best_prediction(predictions)["confidence"] == 0.87


def test_best_prediction_handles_empty_list():
    assert detectors.best_prediction([]) is None


def test_heuristic_verifier_uses_stricter_threshold(monkeypatch):
    monkeypatch.setenv("HEURISTIC_CONFIRM_THRESHOLD", "0.65")

    confirmed = verifier._heuristic_verify(0.70, "Pothole")
    rejected = verifier._heuristic_verify(0.60, "Pothole")

    assert confirmed["confirmed"] is True
    assert confirmed["method"] == "heuristic"
    assert rejected["confirmed"] is False


def test_storage_round_trip(tmp_path, monkeypatch):
    pins_file = tmp_path / "pins.json"

    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(storage, "PINS_FILE", str(pins_file))

    pin = storage.add_pin(
        detector="pothole",
        label="Pothole",
        lat=18.76,
        lon=73.85,
        confidence=0.91,
        reason="Test pin",
        image_path=None,
    )

    pins = storage.get_pins()

    assert pins_file.exists()
    assert len(pins) == 1
    assert pins[0]["id"] == pin["id"]
    assert pins[0]["label"] == "Pothole"
    assert json.loads(pins_file.read_text())[0]["confidence"] == 0.91

def test_run_one_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "test-workspace")
    monkeypatch.setenv("ROBOFLOW_POTHOLE_WORKFLOW_ID", "test-workflow")

    result = detectors._run_one_model("pothole", b"fake-image")

    assert result["predictions"] == []
    assert result["error"] == "ROBOFLOW_API_KEY not configured"


def test_run_one_model_parses_successful_response(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "test-workspace")
    monkeypatch.setenv("ROBOFLOW_POTHOLE_WORKFLOW_ID", "test-workflow")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "output": {
                    "predictions": [
                        {"confidence": 0.82, "class": "pothole"},
                        {"confidence": 0.91, "class": "pothole"},
                    ]
                }
            }

    def fake_post(url, json, timeout):
        assert url == (
            "https://serverless.roboflow.com/"
            "test-workspace/workflows/test-workflow"
        )
        assert json["api_key"] == "test-key"
        assert json["inputs"]["image"]["type"] == "base64"
        assert timeout == 25
        return FakeResponse()

    monkeypatch.setattr(detectors.requests, "post", fake_post)

    result = detectors._run_one_model("pothole", b"fake-image")

    assert result["error"] is None
    assert len(result["predictions"]) == 2
    assert result["predictions"][0]["confidence"] == 0.82
    assert result["predictions"][1]["confidence"] == 0.91


def test_run_one_model_handles_request_failure(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "test-workspace")
    monkeypatch.setenv("ROBOFLOW_POTHOLE_WORKFLOW_ID", "test-workflow")

    def fake_post(url, json, timeout):
        raise detectors.requests.exceptions.Timeout("request timed out")

    monkeypatch.setattr(detectors.requests, "post", fake_post)

    result = detectors._run_one_model("pothole", b"fake-image")

    assert result["predictions"] == []
    assert result["error"] == "request timed out"


def test_run_one_model_handles_http_error(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "test-workspace")
    monkeypatch.setenv("ROBOFLOW_POTHOLE_WORKFLOW_ID", "test-workflow")

    def fake_post(url, json, timeout):
        response = Mock()
        response.text = "Roboflow service unavailable"
        error = detectors.requests.exceptions.HTTPError("500 Server Error")
        error.response = response
        raise error

    monkeypatch.setattr(detectors.requests, "post", fake_post)

    result = detectors._run_one_model("pothole", b"fake-image")

    assert result["predictions"] == []
    assert "500 Server Error" in result["error"]
    assert "Roboflow service unavailable" in result["error"]
