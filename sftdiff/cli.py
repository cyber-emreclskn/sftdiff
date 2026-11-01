import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compare import compare_snapshots
from .demo import create_demo
from .report import render_report
from .snapshot import SnapshotError, load_snapshot


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _check_outputs(outputs, inputs=(), *, force=False):
    resolved = [Path(path).resolve() for path in outputs if path]
    if len(set(resolved)) != len(resolved):
        raise SnapshotError("Output paths must be distinct.")
    protected = {Path(path).resolve() for path in inputs}
    for path in resolved:
        if path in protected:
            raise SnapshotError(f"Output would overwrite an input snapshot: {path}")
        if path.exists() and not force:
            raise SnapshotError(f"Output exists: {path}. Use --force to replace it.")


def _write(path, content, *, force=False):
    with Path(path).open("w" if force else "x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _print_summary(result):
    summary = result["summary"]
    print("SFTDiff | training input comparison")
    print("  " + " | ".join(f"{summary[key]} {key}" for key in ("changed", "unchanged", "added", "removed")))
    print(f"  Effective targets: {summary['before_targets']} -> {summary['after_targets']}")
    print(f"  Records with review warnings: {summary['warning_rows']}")
    print("  Differences are evidence, not automatic correctness failures.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sftdiff", description="See what changed in the tokens your fine-tune learns from.")
    parser.add_argument("--version", action="version", version=f"SFTDiff {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare two captured JSONL snapshots by stable row ID")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--html", metavar="PATH", help="Write a standalone, offline report")
    compare.add_argument("--json", metavar="PATH", help="Write full machine-readable evidence")
    compare.add_argument("--max-rows", type=_positive, default=100, help="Maximum displayed HTML records (default: 100)")
    compare.add_argument("--fail-on-change", action="store_true", help="Exit 1 for changed, added or removed records")
    compare.add_argument("--fail-on-warning", action="store_true", help="Exit 1 if either capture has review warnings")
    compare.add_argument("--force", action="store_true", help="Replace existing output files")
    demo = commands.add_parser("demo", help="Generate clearly marked synthetic snapshots and their report")
    demo.add_argument("--out", default="demo", help="Output directory (default: demo)")
    demo.add_argument("--force", action="store_true", help="Replace existing demo artifacts")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            directory = Path(args.out)
            before_path, after_path = directory / "before.jsonl", directory / "after.jsonl"
            html_path, json_path = directory / "report.html", directory / "comparison.json"
            _check_outputs([before_path, after_path, html_path, json_path], force=args.force)
            directory.mkdir(parents=True, exist_ok=True)
            before, after = create_demo(before_path, after_path, force=args.force)
            result = compare_snapshots(before, after)
            _write(html_path, render_report(result, before_name="before.jsonl", after_name="after.jsonl"), force=args.force)
            _write(json_path, json.dumps(result, indent=2, ensure_ascii=False) + "\n", force=args.force)
            _print_summary(result)
            print(f"  Synthetic demo report: {html_path.resolve()}")
            return 0
        _check_outputs([args.html, args.json], [args.before, args.after], force=args.force)
        result = compare_snapshots(load_snapshot(args.before), load_snapshot(args.after))
        if args.html:
            _write(args.html, render_report(result, before_name=Path(args.before).name,
                                            after_name=Path(args.after).name, max_rows=args.max_rows), force=args.force)
        if args.json:
            _write(args.json, json.dumps(result, ensure_ascii=False, indent=2) + "\n", force=args.force)
        _print_summary(result)
        changed = any(result["summary"][key] for key in ("changed", "added", "removed"))
        return int((args.fail_on_change and changed) or (args.fail_on_warning and result["summary"]["warning_rows"] > 0))
    except (SnapshotError, OSError, UnicodeError) as exc:
        print(f"sftdiff: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
