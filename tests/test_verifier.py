import pytest
from unittest.mock import Mock

from backend import verifier


@pytest.fixture(autouse=True)
def mock_supported_gemini_models(monkeypatch):
    # Keep tests isolated from process-wide cache, pacing, and cooldown state.
    monkeypatch.setattr(verifier, "_model_cache", {})
    monkeypatch.setattr(verifier, "_next_request_at", 0.0)
    monkeypatch.setattr(verifier, "_cooldown_until", 0.0)
    monkeypatch.setattr(verifier, "_consecutive_429s", 0)
    monkeypatch.setenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "0")

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            }
        ]
    }
    monkeypatch.setattr(verifier.requests, "get", lambda *args, **kwargs: response)



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
    assert result["error"].endswith("Gemini request timed out")
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


def test_gemini_verify_caches_model_discovery(monkeypatch):
    model_response = Mock()
    model_response.raise_for_status.return_value = None
    model_response.json.return_value = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            }
        ]
    }
    get_mock = Mock(return_value=model_response)
    monkeypatch.setattr(verifier.requests, "get", get_mock)

    success_response = Mock()
    success_response.status_code = 200
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {
        "candidates": [{
            "content": {"parts": [{
                "text": '{"confirmed": true, "confidence": 0.9, "reason": "Pothole visible"}'
            }]}
        }]
    }
    monkeypatch.setattr(verifier.requests, "post", Mock(return_value=success_response))

    first = verifier._gemini_verify(b"image-1", "Pothole", "cache-test-key", 20)
    second = verifier._gemini_verify(b"image-2", "Pothole", "cache-test-key", 20)

    assert first["error"] is None
    assert second["error"] is None
    assert get_mock.call_count == 1


def test_gemini_429_enters_cooldown_and_prevents_more_requests(monkeypatch):
    model_response = Mock()
    model_response.raise_for_status.return_value = None
    model_response.json.return_value = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            }
        ]
    }
    monkeypatch.setattr(verifier.requests, "get", Mock(return_value=model_response))

    limited_response = Mock()
    limited_response.status_code = 429
    limited_response.headers = {"Retry-After": "30"}
    post_mock = Mock(return_value=limited_response)
    monkeypatch.setattr(verifier.requests, "post", post_mock)

    first = verifier._gemini_verify(b"image-1", "Pothole", "rate-limit-test-key", 20)
    second = verifier._gemini_verify(b"image-2", "Pothole", "rate-limit-test-key", 20)

    assert first["confirmed"] is False
    assert "rate limit reached" in first["error"].lower()
    assert "30 seconds" in first["error"]
    assert second["confirmed"] is False
    assert "rate-limited" in second["error"].lower()
    assert post_mock.call_count == 1


def test_gemini_429_uses_bounded_backoff_when_retry_after_missing(monkeypatch):
    limited_response = Mock()
    limited_response.headers = {}
    delay = verifier._register_rate_limit(limited_response)

    assert delay == verifier._INITIAL_429_COOLDOWN_SECONDS
    assert verifier._cooldown_until > 0
