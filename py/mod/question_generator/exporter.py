import json
from datetime import datetime
from pathlib import Path
from .config import EXPORT_DIR

def export_to_jsonl(data, filename: str | None = None) -> Path:
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"questions_{ts}.jsonl"
    filepath = EXPORT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return filepath