import json
import os
from datetime import datetime


HISTORY_FILE = os.path.join(
    os.path.dirname(__file__),
    "prediction_history.json"
)


def load_history():
    """Load prediction history from JSON file."""

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_prediction(prediction, risk, default_probability):
    """Save a prediction to history."""

    history = load_history()

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prediction": int(prediction),
        "risk": risk,
        "default_probability": round(
            float(default_probability), 4
        )
    }

    history.append(record)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            indent=4
        )

    return record


def get_history():
    """Return all prediction records."""

    return load_history()


def clear_history():
    """Delete all prediction history."""

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump([], file, indent=4)

    return []