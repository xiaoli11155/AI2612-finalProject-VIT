from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict


class CSVLogger:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.file_path.exists() and self.file_path.stat().st_size > 0

    def log(self, row: Dict):
        with self.file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def save_json(payload: Dict, file_path: str):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
