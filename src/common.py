from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'
PROMPTS_DIR = ROOT / 'prompts'


def bootstrap() -> None:
    load_dotenv(ROOT / '.env', override=False)
    ensure_project_dirs()


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        OUTPUTS_DIR,
        OUTPUTS_DIR / 'raw',
        OUTPUTS_DIR / 'parsed',
        OUTPUTS_DIR / 'figures',
        OUTPUTS_DIR / 'preview_pdfs',
        PROMPTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    for file_path in [
        DATA_DIR / 'templates.jsonl',
        DATA_DIR / 'variants.jsonl',
        DATA_DIR / 'sessions.jsonl',
        OUTPUTS_DIR / 'results.jsonl',
    ]:
        if not file_path.exists():
            file_path.write_text('', encoding='utf-8')


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def reset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('', encoding='utf-8')


def normalize_cli_list(values: list[str] | None, default: list[str] | None = None) -> list[str]:
    if not values:
        return default or []
    out: list[str] = []
    for value in values:
        for piece in value.split(','):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


def env_or_default(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)
