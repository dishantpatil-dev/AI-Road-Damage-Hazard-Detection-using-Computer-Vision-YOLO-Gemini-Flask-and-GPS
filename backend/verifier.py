"""
Sends a candidate frame (one that a Roboflow model already flagged) to
Gemini for a second opinion, so a shadow or a wet patch doesn't get pinned
to the map as a "confirmed" pothole.

If GOOGLE_API_KEY isn't set yet, this automatically falls back to a
heuristic verifier: a candidate is auto-confirmed only if Roboflow's own
confidence clears a *second, higher* bar (HEURISTIC_CONFIRM_THRESHOLD).
This is weaker than Gemini's visual check, but keeps the pipeline usable
before you add the Gemini key — swap it in later and verification quality
improves automatically, no other code changes needed.
"""
import base64
import json
import os
import re
import requests

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

PROMPT_TEMPLATE = (
    "You are reviewing a single frame captured from a road-facing vehicle "
    "dashcam. A separate detection model flagged this frame as possibly "
    "containing: {label}.\n\n"
    "Look carefully at the image and decide if this is a genuine, "
    "real-world instance of \"{label}\" that a road authority should act on "
    "(not a shadow, reflection, painted marking, stock photo, or unrelated "
    "object).\n\n"
    "Respond with ONLY a JSON object, no markdown fences, in this exact "
    "shape:\n"
    '{{"confirmed": true or false, "confidence": 0-1 number, '
    '"reason": "one short sentence"}}'
)


def _gemini_verify(image_bytes: bytes, label: str, api_key: str, timeout: int):
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt = PROMPT_TEMPLATE.format(label=label)

    url = f"{GEMINI_BASE_URL}/{model}:generateContent"
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
    try:
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(text)

        if not isinstance(parsed, dict):
            raise ValueError("Gemini response must be a JSON object")

        required_fields = {"confirmed", "confidence", "reason"}
        missing_fields = required_fields - parsed.keys()
        if missing_fields:
            raise ValueError(
                "Gemini response missing required field(s): "
                + ", ".join(sorted(missing_fields))
            )

        confirmed = parsed["confirmed"]
        if not isinstance(confirmed, bool):
            raise ValueError("Gemini response field 'confirmed' must be a boolean")

        confidence = parsed["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("Gemini response field 'confidence' must be a number")

        if not 0 <= confidence <= 1:
            raise ValueError(
                "Gemini response field 'confidence' must be between 0 and 1"
            )

        reason = parsed["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "Gemini response field 'reason' must be a non-empty string"
            )

        result["confirmed"] = confirmed
        result["confidence"] = float(confidence)
        result["reason"] = reason
    except (requests.exceptions.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        result["error"] = str(e)
    return result


def _heuristic_verify(roboflow_confidence: float, label: str):
    """No Gemini key configured — fall back to a stricter confidence bar
    on the Roboflow prediction itself instead of a second AI opinion."""
    threshold = float(os.environ.get("HEURISTIC_CONFIRM_THRESHOLD", 0.65))
    confirmed = roboflow_confidence >= threshold
    reason = (
        f"Auto-confirmed: model confidence {roboflow_confidence:.0%} ≥ "
        f"{threshold:.0%} threshold (no Gemini key set — this is NOT an "
        f"AI-verified confirmation, just a stricter confidence cutoff)"
        if confirmed else
        f"Not auto-confirmed: model confidence {roboflow_confidence:.0%} "
        f"below {threshold:.0%} threshold"
    )
    return {
        "confirmed": confirmed, "confidence": roboflow_confidence,
        "reason": reason, "error": None, "method": "heuristic",
    }


def verify_detection(image_bytes: bytes, label: str, roboflow_confidence: float = 0.0, timeout: int = 20):
    """Confirm/deny a candidate detection. Uses Gemini if GOOGLE_API_KEY is
    set; otherwise falls back to the confidence heuristic above.
    Returns {"confirmed": bool, "confidence": float, "reason": str,
             "error": str|None, "method": "gemini"|"heuristic"}
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key and "your_" not in api_key:
        return _gemini_verify(image_bytes, label, api_key, timeout)
    return _heuristic_verify(roboflow_confidence, label)
