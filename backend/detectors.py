```python
"""
CivicScan detector wrapper.
Calls the single published Roboflow workflow containing all three models.
"""

import base64
import os

import requests

ROBOFLOW_INFER_URL = "https://serverless.roboflow.com"
WORKFLOW_ID = "custom-workflow"


def _find_prediction_lists(obj, depth=0):
    """Find nested prediction lists in a Roboflow workflow response."""
    if depth > 8:
        return []

    found = []

    if isinstance(obj, dict):
        # Prefer the standard Roboflow detection structure.
        predictions = obj.get("predictions")
        if isinstance(predictions, list):
            found.append(predictions)

        for key, value in obj.items():
            if key != "predictions":
                found.extend(_find_prediction_lists(value, depth + 1))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_prediction_lists(item, depth + 1))

    return found


def _get_workflow_predictions(data):
    """Extract predictions from the published workflow output."""
    root = data.get("predictions", data) if isinstance(data, dict) else data

    # The current workflow exposes detections_combine_output.
    if isinstance(root, dict):
        combined = root.get("detections_combine_output")
        if isinstance(combined, dict):
            preds = combined.get("predictions")
            if isinstance(preds, list):
                return preds

        # Support the second combine block if that is the final output.
        combined = root.get("detections_combine_1_output")
        if isinstance(combined, dict):
            preds = combined.get("predictions")
            if isinstance(preds, list):
                return preds

    lists = _find_prediction_lists(root)
    return max(lists, key=len) if lists else []


def run_all_detectors(image_bytes: bytes):
    """
    Run the main workflow once, then group predictions by class label.
    Keeps the result structure expected by app.py.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE")

    detector_labels = {
        "pothole": "Pothole",
        "garbage": "Garbage Area",
        "encroachment": "Road Encroachment",
    }

    if not api_key or not workspace:
        return [
            {
                "detector": key,
                "label": label,
                "predictions": [],
                "error": "ROBOFLOW_API_KEY or ROBOFLOW_WORKSPACE is not configured",
            }
            for key, label in detector_labels.items()
        ]

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = f"{ROBOFLOW_INFER_URL}/{workspace}/workflows/{WORKFLOW_ID}"

    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_b64,
            }
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        predictions = _get_workflow_predictions(data)

        # Map the workflow's class names to CivicScan categories.
        def category_for(class_name):
            name = str(class_name).strip().lower()

            if any(word in name for word in (
                "pothole", "road damage", "crack"
            )):
                return "pothole"

            if any(word in name for word in (
                "garbage", "trash", "rubbish", "litter",
                "waste", "bag", "dump"
            )):
                return "garbage"

            if any(word in name for word in (
                "encroachment", "roadside obstruction",
                "obstruction", "illegal parking"
            )):
                return "encroachment"

            return None

        grouped = {key: [] for key in detector_labels}

        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue

            category = category_for(
                prediction.get("class", prediction.get("label", ""))
            )

            if category:
                grouped[category].append(prediction)

        return [
            {
                "detector": key,
                "label": label,
                "predictions": grouped[key],
                "error": None,
            }
            for key, label in detector_labels.items()
        ]

    except (requests.RequestException, ValueError) as exc:
        message = f"Roboflow workflow request failed: {exc}"
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            message += f" — {exc.response.text[:300]}"

        return [
            {
                "detector": key,
                "label": label,
                "predictions": [],
                "error": message,
            }
            for key, label in detector_labels.items()
        ]


def best_prediction(predictions):
    """Return the highest-confidence prediction."""
    if not predictions:
        return None

    return max(
        predictions,
        key=lambda prediction: float(prediction.get("confidence", 0)),
    )
```
