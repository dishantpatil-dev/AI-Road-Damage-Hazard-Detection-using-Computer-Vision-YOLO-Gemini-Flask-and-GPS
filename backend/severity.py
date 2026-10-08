"""Configurable heuristic severity scoring for confirmed road issues.

This is deliberately a transparent rule-based score, not a trained severity
model. It combines detector confidence with a configurable category weight.
The thresholds and weights can be tuned through environment variables.
"""

import os


DEFAULT_WEIGHTS = {
    "pothole": 1.0,
    "garbage": 1.0,
    "encroachment": 1.0,
}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _weight_for(detector: str, label: str) -> float:
    key = detector.strip().lower().replace(" ", "_")
    label_key = label.strip().lower().replace(" ", "_")

    defaults = {
        **DEFAULT_WEIGHTS,
        "road_encroachment": DEFAULT_WEIGHTS["encroachment"],
        "garbage_area": DEFAULT_WEIGHTS["garbage"],
    }

    default = defaults.get(key, defaults.get(label_key, 1.0))
    return max(0.0, _float_env(f"SEVERITY_WEIGHT_{key.upper()}", default))


def calculate_severity(
    detector: str,
    label: str,
    confidence: float,
) -> dict:
    """Return a transparent severity classification and 0-100 score."""
    confidence = max(0.0, min(1.0, float(confidence)))
    weight = _weight_for(detector, label)

    score = round(min(100.0, confidence * 100.0 * weight), 1)
    medium_threshold = _float_env("SEVERITY_MEDIUM_THRESHOLD", 40.0)
    high_threshold = _float_env("SEVERITY_HIGH_THRESHOLD", 70.0)

    if score >= high_threshold:
        level = "HIGH"
    elif score >= medium_threshold:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "severity": level,
        "severity_score": score,
    }
