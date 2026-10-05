"""Atomic JSON checkpoints; never trust file existence alone."""
import hashlib
import json
import os
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

def completed(path, task):
    try:
        value = json.loads(Path(path).read_text())
        return value.get("task") == task and value.get("status") == "ok"
    except (OSError, ValueError):
        return False
