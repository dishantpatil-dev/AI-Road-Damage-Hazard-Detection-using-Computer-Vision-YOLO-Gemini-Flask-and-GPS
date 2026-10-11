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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL_FALLBACKS = ("gemini-2.5-flash", "gemini-2.0-flash")


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
    configured_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    models_to_try = [configured_model] + [
        model for model in MODEL_FALLBACKS if model != configured_model
    ]
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

    result = {"confirmed": False, "confidence": 0.0, "reason": "", "error": None, "method": "gemini"}
    errors = []
    for model in models_to_try:
        url = f"{GEMINI_BASE_URL}/{model}:generateContent"
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=timeout)
            if resp.status_code == 404:
                errors.append(f"{model}: model not found (404)")
                continue
            resp.raise_for_status()
            result.update(_parse_verdict(resp.json()))
            result["model"] = model
            return result
        except (requests.exceptions.RequestException, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result["error"] = str(exc)
            # Only try another model when the requested model is not found.
            # For auth, quota, network, or malformed output errors, report it.
            return result

    result["error"] = "No configured Gemini model was found: " + "; ".join(errors)
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
