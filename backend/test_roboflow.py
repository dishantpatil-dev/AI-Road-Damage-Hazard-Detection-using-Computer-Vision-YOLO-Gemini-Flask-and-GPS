```python
"""
Test CivicScan's main Roboflow pipeline independently of Flask.

Usage:
    python test_roboflow.py path/to/road_photo.jpg
"""

import base64
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKFLOW_ID = "custom-workflow"

URL = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_ID}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_roboflow.py path/to/image.jpg")
        sys.exit(1)

    if not API_KEY or not WORKSPACE:
        print("Missing ROBOFLOW_API_KEY or ROBOFLOW_WORKSPACE in .env")
        sys.exit(1)

    with open(sys.argv[1], "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "api_key": API_KEY,
        "inputs": {
            "image": {
                "type": "base64",
                "value": encoded_image
            }
        }
    }

    print(f"Testing main workflow: {WORKFLOW_ID}")
    print(f"Workspace: {WORKSPACE}")
    print("Sending image...")

    try:
        response = requests.post(URL, json=payload, timeout=60)

        print(f"\nHTTP Status: {response.status_code}")
        print("Response:")

        try:
            data = response.json()
            print(json.dumps(data, indent=2)[:12000])

            if response.ok:
                predictions = (
                    data.get("predictions", {})
                    .get("detections_combine_output", {})
                    .get("predictions", [])
                )

                print(f"\nCombined detections found: {len(predictions)}")

                for prediction in predictions:
                    print(
                        f"- {prediction.get('class', 'unknown')}: "
                        f"{prediction.get('confidence', 0):.2%} confidence"
                    )

                if not predictions:
                    print(
                        "No predictions at the expected path. "
                        "Inspect the JSON above for the actual response structure."
                    )

        except ValueError:
            print(response.text[:12000])

    except requests.RequestException as error:
        print(f"Request failed: {error}")


if __name__ == "__main__":
    main()
```
