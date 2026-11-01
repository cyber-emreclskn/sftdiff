"""Evidence comparison: differences are not automatically training bugs."""

from difflib import SequenceMatcher

from .snapshot import SnapshotError, validate


def _mask(row):
    return row.get("attention_mask", [1] * len(row["input_ids"]))


def _targets(row):
    # Standard causal LM: logits at i-1 predict labels at i. Label 0 is unused.
    return sum(label != -100 for label in row["labels"][1:])


def _warnings(row):
    warnings = []
    if not _targets(row):
        warnings.append("No effective supervised targets after the standard causal shift.")
    if any(i > 0 and label != -100 and not mask for i, (label, mask) in enumerate(zip(row["labels"], _mask(row)))):
        warnings.append("Padding positions have unmasked labels; review your objective.")
    if any(label != -100 and label != token for label, token in zip(row["labels"], row["input_ids"])):
        warnings.append("Some labels differ from input IDs; verify unshifted causal-LM targets.")
    return warnings


def compare_snapshots(before, after):
    validate(before)
    validate(after)
    if before.header["tokenizer"]["fingerprint"] != after.header["tokenizer"]["fingerprint"]:
        raise SnapshotError(
            "Tokenizer vocab/special-ID fingerprints differ. Token-ID comparison would be misleading. "
            "v0.1 requires a shared token vocabulary."
        )
    left = {row["id"]: row for row in before.rows}
    right = {row["id"]: row for row in after.rows}
    diffs = []
    summary = {"before_rows": len(left), "after_rows": len(right), "changed": 0,
               "unchanged": 0, "added": 0, "removed": 0, "warning_rows": 0,
               "before_targets": sum(_targets(r) for r in left.values()),
               "after_targets": sum(_targets(r) for r in right.values())}
    for row_id in [*left, *(key for key in right if key not in left)]:
        a, b = left.get(row_id), right.get(row_id)
        changes = []
        if a is None or b is None:
            status = "added" if a is None else "removed"
            opcodes = [("insert" if a is None else "delete", 0, len(a["input_ids"]) if a else 0,
                        0, len(b["input_ids"]) if b else 0)]
        else:
            same_ids = a["input_ids"] == b["input_ids"]
            if not same_ids:
                changes.append("tokens")
            if a["labels"] != b["labels"]:
                changes.append("labels")
            if _mask(a) != _mask(b):
                changes.append("attention")
            status = "changed" if changes else "unchanged"
            # Sequence alignment is a presentation aid, not source-position proof.
            opcodes = [("equal", 0, len(a["input_ids"]), 0, len(b["input_ids"]))] if same_ids else list(
                SequenceMatcher(None, a["input_ids"], b["input_ids"], autojunk=True).get_opcodes()
            )
        warnings = {"before": _warnings(a) if a is not None else [],
                    "after": _warnings(b) if b is not None else []}
        summary[status] += 1
        summary["warning_rows"] += int(bool(warnings["before"] or warnings["after"]))
        diffs.append({
            "id": row_id, "status": status, "changes": changes,
            "before": a, "after": b, "opcodes": opcodes, "warnings": warnings,
            "before_targets": _targets(a) if a is not None else 0,
            "after_targets": _targets(b) if b is not None else 0,
        })
    return {"format": "sftdiff.comparison.v1", "summary": summary,
            "before_header": before.header, "after_header": after.header, "rows": diffs}
