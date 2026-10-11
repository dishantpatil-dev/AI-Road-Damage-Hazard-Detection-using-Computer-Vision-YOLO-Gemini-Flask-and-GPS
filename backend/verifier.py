"""
Gemini second-opinion verifier with a safe confidence-based fallback.
If the configured Gemini model is no longer available, try current fallback
models rather than failing every detection.
"""
import base64
import json
import os
import re

import requests

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL_PREFERENCE = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash")


def _list_supported_models(api_key: str, timeout: int):
    """Return models this API key can access that support generateContent."""
    response = requests.get(
        f"{GEMINI_BASE_URL}/models",
        params={"key": api_key},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    models = []
    for item in payload.get("models", []):
        name = item.get("name", "")
        methods = item.get("supportedGenerationMethods", [])
        if name.startswith("models/"):
            name = name[len("models/"):]
        if name and "generateContent" in methods:
            models.append(name)
    return models


def _ordered_models(available_models, configured_model):
    """Try configured model first, then known current models, then remaining listed models."""
    ordered = []
    if configured_model and configured_model in available_models:
        ordered.append(configured_model)
    for model in DEFAULT_MODEL_PREFERENCE:
        if model in available_models and model not in ordered:
            ordered.append(model)
    for model in sorted(available_models, reverse=True):
        if model not in ordered:
            ordered.append(model)
    return ordered


PROMPT_TEMPLATE = (
    "You are reviewing a single frame captured from a road-facing vehicle "
    "dashcam. A separate detection model flagged this frame as possibly "
    "containing: {label}.\n\n"
    "Decide whether the image shows a genuine real-world instance of "
    "{label} that a road authority should act on, not a shadow, reflection, "
    "painted marking, or unrelated object.\n\n"
    "Respond with ONLY a JSON object in this exact shape: "
    '{{"confirmed": true or false, "confidence": 0-1 number, '
    '"reason": "one short sentence"}}'
)


def _parse_verdict(response_data):
    text = response_data["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^\`\`\`(?:json)?\s*|\s*\`\`\`$", "", text.strip(), flags=re.MULTILINE).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object")
    if not isinstance(parsed.get("confirmed"), bool):
        raise ValueError("Gemini response field 'confirmed' must be a boolean")
    confidence = parsed.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Gemini response field 'confidence' must be a number between 0 and 1")
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Gemini response field 'reason' must be a non-empty string")
    return {
        "confirmed": parsed["confirmed"],
        "confidence": float(confidence),
        "reason": reason.strip(),
    }


def _gemini_verify(image_bytes: bytes, label: str, api_key: str, timeout: int):
    configured_model = os.environ.get("GEMINI_MODEL", "").strip()
    result = {"confirmed": False, "confidence": 0.0, "reason": "", "error": None, "method": "gemini"}

    try:
        available_models = _list_supported_models(api_key, timeout)
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        result["error"] = (
            f"Could not list Gemini models (HTTP {status}). Check that the Google AI "
            "API key is valid and the Generative Language API is enabled."
        )
        return result
    except (requests.exceptions.RequestException, ValueError, TypeError) as exc:
        result["error"] = f"Could not retrieve supported Gemini models: {exc}"
        return result

    models_to_try = _ordered_models(available_models, configured_model)
    if not models_to_try:
        result["error"] = (
            "This API key returned no models supporting generateContent. Check the "
            "key's project/API access in Google AI Studio."
        )
        return result

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt = PROMPT_TEMPLATE.format(label=label)
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
            ]
        }],
        "generationConfig": {"temperature": 0.1},
    }
    not_found = []
    for model in models_to_try:
        url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
        try:
            # Header auth prevents the API key from appearing in request URLs/logs.
            resp = requests.post(
                url,
                headers={"x-goog-api-key": api_key},
                json=body,
                timeout=timeout,
            )
            if resp.status_code == 404:
                not_found.append(model)
                continue
            resp.raise_for_status()
            result.update(_parse_verdict(resp.json()))
            result["model"] = model
            if configured_model and model != configured_model:
                result["reason"] = f"Used available model {model}. " + result["reason"]
            return result
        except (requests.exceptions.RequestException, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            # Only model-not-found responses trigger another model attempt.
            result["error"] = f"Gemini verification failed using {model}: {exc}"
            return result

    result["error"] = (
        "All listed Gemini models returned 404 for generateContent: "
        + ", ".join(not_found)
        + ". Check the API key's project/model access in Google AI Studio."
    )
    return result


def _heuristic_verify(roboflow_confidence: float, label: str):
    """Fallback when no Gemini API key is configured."""
    threshold = float(os.environ.get("HEURISTIC_CONFIRM_THRESHOLD", 0.65))
    confirmed = roboflow_confidence >= threshold
    reason = (
        f"Auto-confirmed: model confidence {roboflow_confidence:.0%} ≥ "
        f"{threshold:.0%} threshold (heuristic only; Gemini is not configured)"
        if confirmed else
        f"Not auto-confirmed: model confidence {roboflow_confidence:.0%} "
        f"below {threshold:.0%} threshold"
    )
    return {
        "confirmed": confirmed,
        "confidence": roboflow_confidence,
        "reason": reason,
        "error": None,
        "method": "heuristic",
    }


def verify_detection(image_bytes: bytes, label: str, roboflow_confidence: float = 0.0, timeout: int = 20):
    """Verify one candidate. Uses Gemini when a key is configured, otherwise a heuristic."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key and "your_" not in api_key:
        return _gemini_verify(image_bytes, label, api_key, timeout)
    return _heuristic_verify(roboflow_confidence, label)
