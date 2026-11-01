# Snapshot v1

UTF-8 JSONL, optionally with a BOM. First line is the header; every following line is one source record. Blank lines, duplicate IDs and non-finite JSON numbers are rejected.

```json
{"sftdiff":1,"label_alignment":"unshifted","tokenizer":{"name":"example/tokenizer","fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"metadata":{"packing":false}}
{"id":"source-row-42","input_ids":[10,20,30],"labels":[-100,-100,30],"attention_mask":[1,1,1],"tokens":["<user>","Question","Answer"]}
```

The fingerprint above is a format illustration, not a real tokenizer hash. Prefer `capture_batch` to generate snapshots.

## Header

| Field | Contract |
| --- | --- |
| `sftdiff` | Integer `1` |
| `label_alignment` | Exactly `unshifted` |
| `tokenizer.name` | String identifying the tokenizer |
| `tokenizer.fingerprint` | Lowercase 64-character SHA-256 digest |
| `metadata` | Optional JSON object for framework versions, template revision, max length, etc. A truthy `packing` value is rejected. |

The exporter hashes the UTF-8 encoding of Python's `json.dumps({"vocab": tokenizer.get_vocab(), "special_ids": sorted(set(tokenizer.all_special_ids))}, sort_keys=True, ensure_ascii=False)` with its default separators. This guards the token-ID interpretation, not authenticity. Template/normalizer changes are intentionally not part of this fingerprint so their resulting sequence changes can be compared. Record their identity separately in metadata.

## Record

| Field | Contract |
| --- | --- |
| `id` | Non-empty, unique string identifying the same source example across runs |
| `input_ids` | List of non-negative integers, including special/padding tokens |
| `labels` | Same-length list; each value is a non-negative target ID or `-100` |
| `attention_mask` | Optional same-length list of integer 0/1; absent means all ones |
| `tokens` | Optional same-length list of strings for display; absent displays numeric IDs |

Booleans are not accepted as integer token IDs or masks. Empty sequences are permitted as diagnostic evidence and receive a zero-target warning. Source IDs must come from the data, not the current shuffled batch offset. Do not put multiple packed examples under one row ID.

Targets are counted at positions `1..N-1` whose labels are not `-100`, consistent with standard causal shifting inside the model. An unmasked label at position 0 is still compared, but not counted as an effective target. Padding targets are counted and flagged rather than suppressed.

## Comparison JSON

`format` is `sftdiff.comparison.v1`. `summary` contains row counts and effective-target totals. `before_header`/`after_header` preserve provenance. `rows` includes the actual before/after records, status (`changed`, `unchanged`, `added`, `removed`), change categories, warnings and alignment opcodes.

Opcodes use the `difflib` convention: `(tag, before_start, before_end, after_start, after_end)`, with exclusive ends. They form a valid sequence transformation but are not promised to be a minimal edit script or an unambiguous mapping of repeated tokens. Row ordering and metadata differences are not semantic comparison failures.
