import json

from unittest.mock import Mock

from backend import detectors, storage, verifier


def test_find_predictions_from_nested_workflow_response():
    payload = {
        "output": {
            "predictions": [
                {"confidence": 0.31, "x": 10},
                {"confidence": 0.87, "x": 20},
            ]
        }
    }

    predictions = detectors._get_workflow_predictions(payload)

    assert len(predictions) == 2
    assert detectors.best_prediction(predictions)["confidence"] == 0.87


def test_find_predictions_from_combined_workflow_output():
    payload = {
        "predictions": {
            "detections_combine_output": {
                "predictions": [{"confidence": 0.91, "class": "pothole"}]
            }
        }
    }

    assert detectors._get_workflow_predictions(payload) == [
        {"confidence": 0.91, "class": "pothole"}
    ]


def test_best_prediction_handles_empty_list():
    assert detectors.best_prediction([]) is None


def test_run_all_detectors_requires_api_configuration(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.delenv("ROBOFLOW_WORKSPACE", raising=False)

    results = detectors.run_all_detectors(b"fake-image")

    assert len(results) == 3
    assert all(item["predictions"] == [] for item in results)
    assert all("not configured" in item["error"] for item in results)


def test_run_all_detectors_groups_predictions(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "test-workspace")

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "predictions": {
            "detections_combine_output": {
                "predictions": [
                    {"confidence": 0.82, "class": "pothole"},
                    {"confidence": 0.91, "class": "garbage"},
                    {"confidence": 0.76, "class": "road encroachment"},
                    {"confidence": 0.2, "class": "unknown object"},
                ]
            }
        }
    }

    def fake_post(url, json, timeout):
        assert url == (
            "https://serverless.roboflow.com/"
            "test-workspace/workflows/custom-workflow"
        )
        assert json["api_key"] == "test-key"
        assert json["inputs"]["image"]["type"] == "base64"
        assert timeout == 45
        return response

    monkeypatch.setattr(detectors.requests, "post", fake_post)

    results = detectors.run_all_detectors(b"fake-image")
    by_detector = {item["detector"]: item for item in results}

    assert by_detector["pothole"]["predictions"][0]["confidence"] == 0.82
    assert by_detector["garbage"]["predictions"][0]["confidence"] == 0.91
    assert by_detector["encroachment"]["predictions"][0]["confidence"] == 0.76
    assert all(item["error"] is None for item in results)


def test_run_all_detectors_handles_request_failure(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "test-workspace")

    def fake_post(url, json, timeout):
        raise detectors.requests.exceptions.Timeout("request timed out")

    monkeypatch.setattr(detectors.requests, "post", fake_post)

    results = detectors.run_all_detectors(b"fake-image")

    assert len(results) == 3
    assert all(item["predictions"] == [] for item in results)
    assert all("request timed out" in item["error"] for item in results)


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
