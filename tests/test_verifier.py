from unittest.mock import Mock

from backend import verifier


def test_gemini_verify_parses_successful_response(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"confirmed": true, '
                                '"confidence": 0.94, '
                                '"reason": "Genuine road encroachment"}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        verifier.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Road Encroachment",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is True
    assert result["confidence"] == 0.94
    assert result["reason"] == "Genuine road encroachment"
    assert result["error"] is None
    assert result["method"] == "gemini"


def test_gemini_verify_handles_markdown_json(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "```json\n"
                                '{"confirmed": false, '
                                '"confidence": 0.20, '
                                '"reason": "Only a shadow"}'
                                "\n```"
                            )
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        verifier.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Pothole",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is False
    assert result["confidence"] == 0.20
    assert result["reason"] == "Only a shadow"
    assert result["error"] is None


def test_gemini_verify_handles_invalid_json(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "this is not valid json"}
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        verifier.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Pothole",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is False
    assert result["confidence"] == 0.0
    assert result["error"] is not None
    assert result["method"] == "gemini"


def test_gemini_verify_handles_request_failure(monkeypatch):
    def fake_post(*args, **kwargs):
        raise verifier.requests.exceptions.Timeout("Gemini request timed out")

    monkeypatch.setattr(
        verifier.requests,
        "post",
        fake_post,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Pothole",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is False
    assert result["confidence"] == 0.0
    assert result["error"] == "Gemini request timed out"
    assert result["method"] == "gemini"


def test_verify_detection_uses_gemini_when_api_key_exists(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

    expected = {
        "confirmed": True,
        "confidence": 0.92,
        "reason": "Confirmed by Gemini",
        "error": None,
        "method": "gemini",
    }

    monkeypatch.setattr(
        verifier,
        "_gemini_verify",
        lambda image_bytes, label, api_key, timeout: expected,
    )

    result = verifier.verify_detection(
        b"fake-image",
        "Pothole",
        roboflow_confidence=0.80,
    )

    assert result == expected


def test_gemini_verify_rejects_invalid_confirmed_type(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"confirmed": "false", '
                                '"confidence": 0.20, '
                                '"reason": "Only a shadow"}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        verifier.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Pothole",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is False
    assert result["error"] is not None


def test_gemini_verify_rejects_confidence_out_of_range(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"confirmed": true, '
                                '"confidence": 1.5, '
                                '"reason": "Genuine pothole"}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        verifier.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Pothole",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is False
    assert result["confidence"] == 0.0
    assert result["error"] is not None


def test_gemini_verify_rejects_missing_required_field(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"confirmed": true, '
                                '"confidence": 0.90}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        verifier.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Pothole",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is False
    assert result["confidence"] == 0.0
    assert result["error"] is not None


def test_gemini_verify_rejects_invalid_reason_type(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"confirmed": true, '
                                '"confidence": 0.90, '
                                '"reason": 123}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        verifier.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Pothole",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is False
    assert result["confidence"] == 0.0
    assert result["error"] is not None


def test_gemini_verify_rejects_non_object_json(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "[]"
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        verifier.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = verifier._gemini_verify(
        b"fake-image",
        "Pothole",
        "test-api-key",
        20,
    )

    assert result["confirmed"] is False
    assert result["confidence"] == 0.0
    assert result["error"] is not None