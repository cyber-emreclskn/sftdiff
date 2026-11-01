"""Capture a real, already collated batch. No torch import or retokenization."""

import hashlib
import json

from .snapshot import Snapshot, SnapshotError, validate, write_snapshot


def _lists(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def capture_batch(path, batch, tokenizer, *, row_ids, metadata=None, force=False):
    """Save an unpacked batch with caller-provided, stable source row IDs.

    Call after the *actual* collator. Do not infer IDs from shuffled batch indices.
    Only standard, unshifted causal-LM labels are supported. Inputs stay intact.
    """
    if "input_ids" not in batch or "labels" not in batch:
        raise SnapshotError("Capture requires input_ids and labels from the final collator.")
    matrices = {key: _lists(batch[key]) for key in ("input_ids", "labels")}
    if "attention_mask" in batch:
        matrices["attention_mask"] = _lists(batch["attention_mask"])
    if any(not isinstance(value, list) for value in matrices.values()):
        raise SnapshotError("Capture expects two-dimensional token batches.")
    count = len(matrices["input_ids"])
    if isinstance(row_ids, (str, bytes)):
        raise SnapshotError("row_ids must be a sequence of stable IDs, not a string.")
    try:
        row_ids = list(row_ids)
    except TypeError as exc:
        raise SnapshotError("row_ids must be a sequence of stable IDs.") from exc
    if count != len(row_ids) or any(len(v) != count for v in matrices.values()):
        raise SnapshotError("Each batch row needs exactly one stable row ID and matching labels.")
    if any(not isinstance(row, list) for values in matrices.values() for row in values):
        raise SnapshotError("Capture expects two-dimensional, unpacked token batches.")
    if "attention_mask" in batch and any(
        not isinstance(v, int) or isinstance(v, bool) for row in matrices["attention_mask"] for v in row
    ):
        raise SnapshotError("Capture expects a 2D integer attention mask, not an attention matrix.")
    try:
        vocabulary = tokenizer.get_vocab()
        identity = {
            "vocab": vocabulary,
            "special_ids": sorted(set(tokenizer.all_special_ids)),
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    except (AttributeError, TypeError, ValueError) as exc:
        raise SnapshotError("Tokenizer must expose get_vocab() and all_special_ids.") from exc
    rows = []
    for index, row_id in enumerate(row_ids):
        row = {"id": row_id}
        row.update({key: values[index] for key, values in matrices.items()})
        rows.append(row)
    snapshot = Snapshot(
        {
            "sftdiff": 1,
            "label_alignment": "unshifted",
            "tokenizer": {
                "name": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                "fingerprint": fingerprint,
            },
            "metadata": metadata or {},
        },
        rows,
    )
    validate(snapshot)
    for row in snapshot.rows:
        try:
            tokens = tokenizer.convert_ids_to_tokens(row["input_ids"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SnapshotError(f"Tokenizer could not name captured token IDs for {row['id']!r}.") from exc
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise SnapshotError("Tokenizer must provide a token string for every captured token ID.")
        row["tokens"] = tokens
    write_snapshot(path, snapshot, force=force)
    return snapshot
