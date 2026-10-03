import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app


def test_detector_error_is_returned(monkeypatch):
    detector_result = [
        {
            "detector": "pothole",
            "label": "Pothole",
            "predictions": [],
            "error": "Roboflow unavailable",
        }
    ]

    monkeypatch.setattr(
        app.detectors,
        "run_all_detectors",
        lambda image: detector_result,
    )

    result = app.analyze_frame(b"fake-image", 18.76, 73.85)

    assert result[0]["status"] == "error"
    assert result[0]["detector"] == "pothole"
    assert result[0]["message"] == "Roboflow unavailable"


def test_low_confidence_detection_is_clear(monkeypatch):
    detector_result = [
        {
            "detector": "pothole",
            "label": "Pothole",
            "predictions": [
                {"confidence": 0.20}
            ],
            "error": None,
        }
    ]

    def fail_verification(*args, **kwargs):
        raise AssertionError("Verification should not run")

    monkeypatch.setattr(
        app.detectors,
        "run_all_detectors",
        lambda image: detector_result,
    )
    monkeypatch.setattr(
        app.verifier,
        "verify_detection",
        fail_verification,
    )

    result = app.analyze_frame(b"fake-image", 18.76, 73.85)

    assert result[0]["status"] == "clear"
    assert result[0]["message"] == "No candidate above threshold"


def test_rejected_detection_is_not_stored(monkeypatch):
    detector_result = [
        {
            "detector": "pothole",
            "label": "Pothole",
            "predictions": [
                {"confidence": 0.80}
            ],
            "error": None,
        }
    ]

    verdict = {
        "confirmed": False,
        "confidence": 0.30,
        "reason": "Not a genuine pothole",
        "error": None,
        "method": "mock",
    }

    def fail_storage(**kwargs):
        raise AssertionError("Rejected detection must not be stored")

    monkeypatch.setattr(
        app.detectors,
        "run_all_detectors",
        lambda image: detector_result,
    )
    monkeypatch.setattr(
        app.verifier,
        "verify_detection",
        lambda image, label, roboflow_confidence: verdict,
    )
    monkeypatch.setattr(
        app.storage,
        "add_pin",
        fail_storage,
    )

    result = app.analyze_frame(b"fake-image", 18.76, 73.85)

    assert result[0]["status"] == "rejected"
    assert result[0]["message"] == "[mock] Not a genuine pothole"
    assert result[0]["roboflow_confidence"] == 0.80


def test_confirmed_detection_is_stored(monkeypatch):
    detector_result = [
        {
            "detector": "pothole",
            "label": "Pothole",
            "predictions": [
                {"confidence": 0.80}
            ],
            "error": None,
        }
    ]

    verdict = {
        "confirmed": True,
        "confidence": 0.91,
        "reason": "Genuine pothole",
        "error": None,
        "method": "mock",
    }

    expected_pin = {
        "id": "test-pin",
        "type": "pothole",
        "label": "Pothole",
        "lat": 18.76,
        "lon": 73.85,
        "confidence": 0.91,
        "reason": "Genuine pothole",
        "image": "/uploads/test.jpg",
    }

    monkeypatch.setattr(
        app.detectors,
        "run_all_detectors",
        lambda image: detector_result,
    )

    monkeypatch.setattr(
        app.verifier,
        "verify_detection",
        lambda image, label, roboflow_confidence: verdict,
    )

    monkeypatch.setattr(
        app,
        "_save_frame",
        lambda image: "/uploads/test.jpg",
    )

    storage_call = {}


    def fake_add_pin(**kwargs):
        storage_call.update(kwargs)
        return expected_pin


    monkeypatch.setattr(
        app.storage,
        "add_pin",
        fake_add_pin,
    )
    result = app.analyze_frame(b"fake-image", 18.76, 73.85)

    assert result[0]["status"] == "confirmed"
    assert result[0]["pin"] == expected_pin

    assert storage_call == {
        "detector": "pothole",
        "label": "Pothole",
        "lat": 18.76,
        "lon": 73.85,
        "confidence": 0.91,
        "reason": "Genuine pothole",
        "image_path": "/uploads/test.jpg",
    }


def test_confirmed_detection_without_location_is_not_stored(monkeypatch):
    detector_result = [
        {
            "detector": "pothole",
            "label": "Pothole",
            "predictions": [
                {"confidence": 0.80}
            ],
            "error": None,
        }
    ]

    verdict = {
        "confirmed": True,
        "confidence": 0.91,
        "reason": "Genuine pothole",
        "error": None,
        "method": "mock",
    }

    def fail_storage(**kwargs):
        raise AssertionError("Pin should not be stored without GPS")

    monkeypatch.setattr(
        app.detectors,
        "run_all_detectors",
        lambda image: detector_result,
    )

    monkeypatch.setattr(
        app.verifier,
        "verify_detection",
        lambda image, label, roboflow_confidence: verdict,
    )

    monkeypatch.setattr(
        app.storage,
        "add_pin",
        fail_storage,
    )

    result = app.analyze_frame(b"fake-image", None, None)

    assert result[0]["status"] == "error"
    assert "no location available" in result[0]["message"]


def test_multiple_detectors_are_processed_independently(monkeypatch):
    detector_results = [
        {
            "detector": "pothole",
            "label": "Pothole",
            "predictions": [{"confidence": 0.20}],
            "error": None,
        },
        {
            "detector": "garbage",
            "label": "Garbage Area",
            "predictions": [{"confidence": 0.80}],
            "error": None,
        },
        {
            "detector": "encroachment",
            "label": "Road Encroachment",
            "predictions": [{"confidence": 0.90}],
            "error": None,
        },
    ]

    verifier_calls = []

    def fake_verify(image, label, roboflow_confidence):
        verifier_calls.append(
            (label, roboflow_confidence)
        )

        if label == "Garbage Area":
            return {
                "confirmed": False,
                "confidence": 0.20,
                "reason": "Not confirmed",
                "error": None,
                "method": "mock",
            }

        return {
            "confirmed": True,
            "confidence": 0.95,
            "reason": "Confirmed",
            "error": None,
            "method": "mock",
        }

    monkeypatch.setattr(
        app.detectors,
        "run_all_detectors",
        lambda image: detector_results,
    )
    monkeypatch.setattr(
        app.verifier,
        "verify_detection",
        fake_verify,
    )
    monkeypatch.setattr(
        app,
        "_save_frame",
        lambda image: "/uploads/test.jpg",
    )
    monkeypatch.setattr(
        app.storage,
        "add_pin",
        lambda **kwargs: {
            "id": "test-pin",
            **kwargs,
        },
    )

    result = app.analyze_frame(
        b"fake-image",
        18.76,
        73.85,
    )

    assert len(result) == 3

    assert result[0]["detector"] == "pothole"
    assert result[0]["status"] == "clear"

    assert result[1]["detector"] == "garbage"
    assert result[1]["status"] == "rejected"

    assert result[2]["detector"] == "encroachment"
    assert result[2]["status"] == "confirmed"

    assert verifier_calls == [
        ("Garbage Area", 0.80),
        ("Road Encroachment", 0.90),
    ]