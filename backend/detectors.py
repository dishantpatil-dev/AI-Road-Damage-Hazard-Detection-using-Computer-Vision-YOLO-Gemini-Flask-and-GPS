"""
CivicScan detector wrapper.
Calls the single published Roboflow workflow containing the road-issue models.
The returned shape matches the interface expected by backend/app.py.
"""
import base64
import os

import requests

ROBOFLOW_INFER_URL = "https://serverless.roboflow.com"
WORKFLOW_ID = "custom-workflow"

DETECTOR_LABELS = {
    "pothole": "Pothole",
    "garbage": "Garbage Area",
    "encroachment": "Road Encroachment",
}


def _get_workflow_predictions(data):
    """Extract the combined prediction list from Roboflow workflow output."""
    root = data.get("predictions", data) if isinstance(data, dict) else data
    if isinstance(root, dict):
        # Prefer known combine outputs, including the alternate block name.
        for key in ("detections_combine_output", "detections_combine_1_output"):
            block = root.get(key)
            if isinstance(block, dict) and isinstance(block.get("predictions"), list):
                return block["predictions"]

    # Fallback: recursively find prediction lists and use the largest.
    candidates = []

    def walk(obj, depth=0):
        if depth > 8:
            return
        if isinstance(obj, dict):
            preds = obj.get("predictions")
            if isinstance(preds, list) and all(
                isinstance(item, dict) and "confidence" in item for item in preds
            ):
                candidates.append(preds)
            for key, value in obj.items():
                if key != "predictions":
                    walk(value, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, depth + 1)

    walk(root)
    return max(candidates, key=len) if candidates else []


def _category_for(prediction):
    """Map model class names to CivicScan's three supported categories."""
    name = str(prediction.get("class", prediction.get("label", ""))).strip().lower()
    normalized = name.replace("_", " ").replace("-", " ")

    if any(term in normalized for term in (
        "pothole", "road damage", "road crack", "crack"
    )):
        return "pothole"
    if any(term in normalized for term in (
        "garbage", "trash", "rubbish", "litter", "waste", "bag", "dump", "debris"
    )):
        return "garbage"
    if any(term in normalized for term in (
        "encroachment", "roadside obstruction", "obstruction", "illegal parking",
        "road blockage", "blocked road"
    )):
        return "encroachment"
    return None


def run_all_detectors(image_bytes: bytes):
    """Run the published workflow once and group its predictions by category."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE")

    if not api_key or not workspace:
        error = "ROBOFLOW_API_KEY or ROBOFLOW_WORKSPACE is not configured"
        return [
            {"detector": key, "label": label, "predictions": [], "error": error}
            for key, label in DETECTOR_LABELS.items()
        ]

    url = f"{ROBOFLOW_INFER_URL}/{workspace}/workflows/{WORKFLOW_ID}"
    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": base64.b64encode(image_bytes).decode("utf-8"),
            }
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        predictions = _get_workflow_predictions(data)
        grouped = {key: [] for key in DETECTOR_LABELS}

        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            category = _category_for(prediction)
            if category:
                try:
                    prediction["confidence"] = float(prediction.get("confidence", 0))
                except (TypeError, ValueError):
                    continue
                grouped[category].append(prediction)

        return [
            {
                "detector": key,
                "label": label,
                "predictions": grouped[key],
                "error": None,
            }
            for key, label in DETECTOR_LABELS.items()
        ]

    except (requests.RequestException, ValueError) as exc:
        detail = str(exc)
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            detail += f" — {exc.response.text[:300]}"
        error = f"Roboflow workflow request failed: {detail}"
        return [
            {"detector": key, "label": label, "predictions": [], "error": error}
            for key, label in DETECTOR_LABELS.items()
        ]


def best_prediction(predictions):
    """Return the highest-confidence prediction, or None for an empty list."""
    if not predictions:
        return None
    return max(predictions, key=lambda item: float(item.get("confidence", 0)))
