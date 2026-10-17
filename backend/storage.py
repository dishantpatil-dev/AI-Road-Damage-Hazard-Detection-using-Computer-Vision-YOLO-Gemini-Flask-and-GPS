"""
Minimal storage layer for confirmed pins. Uses a JSON file so the project
runs with zero database setup. Swap this out for Postgres/PostGIS later
without touching app.py's calling code (same 3 functions).
"""
import json
import os
import threading
import time
import uuid
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PINS_FILE = os.path.join(DATA_DIR, "pins.json")
_lock = threading.Lock()


def _ensure_store():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PINS_FILE):
        with open(PINS_FILE, "w") as f:
            json.dump([], f)


def add_pin(detector: str, label: str, lat: float, lon: float,
            confidence: float, reason: str, image_path: Optional[str]):
    _ensure_store()
    pin = {
        "id": str(uuid.uuid4()),
        "type": detector,
        "label": label,
        "lat": lat,
        "lon": lon,
        "confidence": round(confidence, 3),
        "reason": reason,
        "image": image_path,
        "timestamp": time.time(),
    }
    with _lock:
        with open(PINS_FILE, "r") as f:
            pins = json.load(f)
        pins.append(pin)
        with open(PINS_FILE, "w") as f:
            json.dump(pins, f, indent=2)
    return pin


def get_pins():
    _ensure_store()
    with _lock:
        with open(PINS_FILE, "r") as f:
            return json.load(f)



def clear_pins():
    """Remove all saved pin records and return how many were removed."""
    _ensure_store()
    with _lock:
        with open(PINS_FILE, "r") as f:
            pins = json.load(f)
        with open(PINS_FILE, "w") as f:
            json.dump([], f, indent=2)
    return len(pins)
