"""The portable, deliberately small snapshot format."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


class SnapshotError(ValueError):
    """Malformed or incompatible training evidence."""


@dataclass
class Snapshot:
    header: dict
    rows: list[dict]


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate(snapshot: Snapshot) -> None:
    header = snapshot.header
    if not isinstance(header, dict) or type(header.get("sftdiff")) is not int or header.get("sftdiff") != 1:
        raise SnapshotError("First line must contain a sftdiff: 1 snapshot header.")
    if header.get("label_alignment") != "unshifted":
        raise SnapshotError("v0.1 requires unshifted causal-LM labels (the HF convention).")
    if not isinstance(header.get("metadata", {}), dict):
        raise SnapshotError("Header metadata must be an object.")
    if header.get("metadata", {}).get("packing"):
        raise SnapshotError("Packed examples are outside v0.1; capture unpacked, stable-ID rows.")
    tokenizer = header.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("name"), str):
        raise SnapshotError("Header must identify the tokenizer by name and fingerprint.")
    fingerprint = tokenizer.get("fingerprint", "")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
        raise SnapshotError("Tokenizer fingerprint must be a SHA-256 hex digest.")
    seen = set()
    for index, row in enumerate(snapshot.rows, 2):
        prefix = f"Record {index}"
        if not isinstance(row, dict):
            raise SnapshotError(f"{prefix}: expected a JSON object.")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise SnapshotError(f"{prefix}: id must be a non-empty, stable string.")
        if row_id in seen:
            raise SnapshotError(f"{prefix}: duplicate id {row_id!r}.")
        seen.add(row_id)
        ids, labels = row.get("input_ids"), row.get("labels")
        if not isinstance(ids, list) or not all(_integer(t) and t >= 0 for t in ids):
            raise SnapshotError(f"{prefix}: input_ids must be non-negative integers.")
        if not isinstance(labels, list) or len(labels) != len(ids):
            raise SnapshotError(f"{prefix}: labels must have the same length as input_ids.")
        if not all(_integer(t) and (t == -100 or t >= 0) for t in labels):
            raise SnapshotError(f"{prefix}: labels must be token IDs or -100.")
        for key in ("tokens", "attention_mask"):
            value = row.get(key)
            if key in row and (not isinstance(value, list) or len(value) != len(ids)):
                raise SnapshotError(f"{prefix}: {key} must match the input_ids length.")
        if "tokens" in row and not all(isinstance(t, str) for t in row["tokens"]):
            raise SnapshotError(f"{prefix}: tokens must be strings.")
        if "attention_mask" in row and not all(
            _integer(t) and t in (0, 1) for t in row["attention_mask"]
        ):
            raise SnapshotError(f"{prefix}: attention_mask must contain integer 0 or 1.")
    try:
        json.dumps(header, allow_nan=False)
        for row in snapshot.rows:
            json.dumps(row, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"Snapshot must contain finite, JSON-serializable data: {exc}") from exc


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def load_snapshot(path) -> Snapshot:
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise SnapshotError(f"{path}:{line_number}: blank JSONL line.")
            try:
                records.append(json.loads(line, parse_constant=_reject_constant))
            except ValueError as exc:
                raise SnapshotError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not records:
        raise SnapshotError(f"{path}: snapshot is empty.")
    snapshot = Snapshot(records[0], records[1:])
    validate(snapshot)
    return snapshot


def write_snapshot(path, snapshot: Snapshot, *, force=False) -> None:
    validate(snapshot)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        for record in [snapshot.header, *snapshot.rows]
    )
    with Path(path).open("w" if force else "x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
