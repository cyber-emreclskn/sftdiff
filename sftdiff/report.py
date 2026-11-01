"""A self-contained, escaped report; no CDN, server, or model calls."""

import base64
import hashlib
import html
import json
from pathlib import Path


def _e(value):
    return html.escape(str(value), quote=True)


def _positions(row, peer, opcodes, side, budget=320):
    """Show changes plus bounded context instead of writing a giant HTML file."""
    size = len(row["input_ids"])
    changed = set()
    attention = row.get("attention_mask", [1] * size)
    peer_attention = peer.get("attention_mask", [1] * len(peer["input_ids"])) if peer is not None else []
    for tag, a0, a1, b0, b1 in opcodes:
        start, end = (a0, a1) if side == "before" else (b0, b1)
        if tag != "equal":
            changed.update(range(start, min(end, start + budget)))
        elif peer is not None:
            for offset in range(end - start):
                i = start + offset
                j = (b0 if side == "before" else a0) + offset
                am = attention[i]
                bm = peer_attention[j]
                if row["labels"][i] != peer["labels"][j] or am != bm:
                    changed.add(i)
                    if len(changed) >= budget:
                        break
    if size <= budget:
        return set(range(size)), changed
    visible = set(range(min(12, size))) | set(range(max(0, size - 12), size))
    for index in sorted(changed):
        if len(visible) >= budget:
            break
        visible.update(range(max(0, index - 4), min(size, index + 5)))
    # Cap after collecting context; keep the changed region and start/end cues.
    return set(sorted(visible)[:budget]), changed


def _panel(row, peer, opcodes, side):
    if row is None:
        return '<div class="empty">No record with this ID</div>'
    visible, changed = _positions(row, peer, opcodes, side)
    pieces, previous = [], -1
    attention = row.get("attention_mask", [1] * len(row["input_ids"]))
    for index in sorted(visible):
        if index > previous + 1:
            pieces.append(f'<span class="gap">… {index - previous - 1} tokens omitted …</span>')
        label = row["labels"][index]
        token_id = row["input_ids"][index]
        active = index > 0 and label != -100
        classes = ["token", "learned" if active else "masked"]
        if index in changed:
            classes.append("delta")
        if not attention[index]:
            classes.append("padding")
        text = row.get("tokens", [])[index] if "tokens" in row else f"#{token_id}"
        tip = f"position {index} | input {token_id} | label {label} | attention {attention[index]}"
        if index == 0:
            tip += " | label at position 0 is unused by the standard causal shift"
        pieces.append(f'<span class="{" ".join(classes)}" title="{_e(tip)}">{_e(text)}</span>')
        previous = index
    if previous < len(row["input_ids"]) - 1:
        pieces.append('<span class="gap">… remaining tokens omitted …</span>')
    if not pieces:
        pieces.append('<span class="gap">Empty sequence</span>')
    return '<div class="token-stream">' + "".join(pieces) + '</div>'


def render_report(result, *, before_name="Before", after_name="After", max_rows=100):
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    root = Path(__file__).parent
    css = (root / "report.css").read_text(encoding="utf-8")
    script = (root / "report.js").read_text(encoding="utf-8")
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    summary = result["summary"]
    important = [r for r in result["rows"] if r["status"] != "unchanged" or any(r["warnings"].values())]
    quiet = [r for r in result["rows"] if r["status"] == "unchanged" and not any(r["warnings"].values())]
    rows = (important + quiet)[:max_rows]
    cards = []
    for number, row in enumerate(rows):
        badges = ''.join(f'<span class="badge">{_e(kind)}</span>' for kind in row["changes"])
        warning_text = ''.join(
            f'<p class="warning"><b>{_e(side)}:</b> {_e(message)}</p>'
            for side, messages in row["warnings"].items() for message in messages
        )
        a = _panel(row["before"], row["after"], row["opcodes"], "before")
        b = _panel(row["after"], row["before"], row["opcodes"], "after")
        cards.append(f'''<details class="record" data-status="{row['status']}" data-id="{_e(row['id'].lower())}" {'open' if number == 0 else ''}>
<summary><span class="status {row['status']}">{row['status']}</span><strong>{_e(row['id'])}</strong>{badges}<span class="target-count">targets {row['before_targets']} → {row['after_targets']}</span></summary>
{warning_text}<div class="sides"><section><h3>BEFORE</h3>{a}</section><section><h3>AFTER</h3>{b}</section></div></details>''')
    metadata = _e(json.dumps({"before": result["before_header"], "after": result["after_header"]}, indent=2, ensure_ascii=False))
    demo = any(h.get("metadata", {}).get("synthetic") for h in (result["before_header"], result["after_header"]))
    demo_note = '<p class="demo-note">SYNTHETIC DEMO · Hand-authored token evidence. This is not a measured model run.</p>' if demo else ''
    metrics = ''.join(f'<div class="metric"><b>{summary[key]:,}</b><span>{label}</span></div>' for key, label in
                      [("changed", "changed records"), ("unchanged", "unchanged"), ("added", "added"), ("removed", "removed"), ("warning_rows", "records to review")])
    shown = f"Showing {len(rows):,} of {len(result['rows']):,} records. Changed and warning records first. Summary covers all captured records."
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{digest}'; base-uri 'none'; form-action 'none'">
<title>SFTDiff · Training input comparison</title><style>{css}</style></head>
<body><main><header><div class="wordmark">SFT<span>DIFF</span> <small>LOCAL EVIDENCE REPORT / v0.1</small></div>
<h1>Same data.<br><em>Different training?</em></h1><p class="intro">See exactly which tokens and learning targets changed before your next fine-tune.</p>
{demo_note}<div class="files"><span>BEFORE <b>{_e(before_name)}</b></span><span>AFTER <b>{_e(after_name)}</b></span></div></header>
<div class="metrics">{metrics}</div><div class="target-bar">Effective targets <b>{summary['before_targets']:,} → {summary['after_targets']:,}</b><span>Unshifted labels · position 0 excluded · counts include unmasked padding</span></div>
<div class="toolbar"><label>Find record <input id="search" placeholder="Search by stable ID" type="search"></label><label>Show <select id="filter"><option value="all">All records</option><option value="changed">Changed</option><option value="added">Added</option><option value="removed">Removed</option><option value="unchanged">Unchanged</option></select></label><button id="expand" type="button">Expand all</button></div>
<div class="legend"><span class="key learned-key">Learning target</span><span class="key masked-key">Masked / unused first label</span><span class="key delta-key">Changed region</span><span class="key padding-key">Padding</span></div>
<p class="caption">{shown} Hover a token for its position, ID and label. Long sequences show bounded context.</p>
<div id="records">{''.join(cards) or '<p class="empty">No records in either snapshot.</p>'}</div><p id="no-results" hidden>No displayed records match this filter.</p>
<aside><b>Evidence, not a correctness verdict.</b> Differences can be intentional. Token alignment uses a sequence-diff heuristic; repeated spans can be ambiguous. Row order is ignored; rows match by ID. Packing, shifted labels, multimodal inputs, numerical loss and gradients are outside v0.1.</aside>
<details class="provenance"><summary>Snapshot provenance</summary><pre>{metadata}</pre></details>
<footer>Generated by SFTDiff · Standalone HTML · No external scripts, fonts or network requests</footer></main><script>{script}</script></body></html>'''
