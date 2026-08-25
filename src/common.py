"""Shared input, masking, and checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def document_text(row: dict[str, Any]) -> str:
    source_info = row.get("source_info")
    if isinstance(source_info, dict):
        for key in ("passages", "document", "article", "context"):
            value = source_info.get(key)
            if isinstance(value, str):
                return value
    if isinstance(source_info, str):
        return source_info
    for key in ("document", "article", "context", "input"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def overlaps_hallucination(labels: list[dict[str, Any]], start: int, end: int) -> int:
    return int(any(max(start, int(x["start"])) < min(end, int(x["end"])) for x in labels))


def mask_span(text: str, start: int, end: int, mask_token: str, count: int) -> str:
    """Mask a character span while preserving tokenizer word boundaries."""
    masks = " ".join([mask_token] * count)
    if start > 0 and not text[start - 1].isspace():
        masks = " " + masks
    if end < len(text) and not text[end].isspace():
        masks += " "
    return text[:start] + masks + text[end:]


def load_encoder_checkpoint(model: torch.nn.Module, checkpoint: str | Path) -> torch.nn.Module:
    """Load either a HF directory or the state dict format saved by this repo."""
    checkpoint = Path(checkpoint)
    state_path = checkpoint / "pytorch_model.bin" if checkpoint.is_dir() else checkpoint
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    target = model.state_dict()
    cleaned = {}
    for key, value in state.items():
        for prefix in ("base_model.", "encoder.", "module.encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key in target and target[key].shape == value.shape:
            cleaned[key] = value
    if not cleaned:
        raise ValueError(f"No compatible encoder weights found in {state_path}")
    model.load_state_dict(cleaned, strict=False)
    return model

