import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from html.parser import HTMLParser
from pathlib import Path

from sftdiff import Snapshot, SnapshotError, capture_batch, compare_snapshots, load_snapshot, write_snapshot
from sftdiff.cli import main
from sftdiff.demo import DemoTokenizer
from sftdiff.report import render_report


def fixture():
    return Snapshot({"sftdiff": 1, "label_alignment": "unshifted",
                     "tokenizer": {"name": "test", "fingerprint": "a" * 64}},
                    [{"id": "a", "input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4],
                      "tokens": ["user", "question", "answer", "eos"], "attention_mask": [1] * 4}])


class SnapshotTests(unittest.TestCase):
    def test_round_trip_unicode(self):
        data = fixture()
        data.rows[0]["id"] = "sipariş-çığ"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.jsonl"
            write_snapshot(path, data)
            self.assertEqual(load_snapshot(path), data)
            with self.assertRaises(FileExistsError):
                write_snapshot(path, data)

    def test_rejects_bad_records(self):
        mutations = [
            lambda s: s.rows[0].update(labels=[-100]),
            lambda s: s.rows[0].update(input_ids=[True, 2, 3, 4]),
            lambda s: s.rows[0].update(labels=[-100, -1, 3, 4]),
            lambda s: s.rows[0].update(tokens=[None] * 4),
            lambda s: s.rows[0].update(tokens=None),
            lambda s: s.rows[0].update(attention_mask=[1, 1, 1, 2]),
            lambda s: s.rows[0].update(attention_mask=None),
            lambda s: s.rows[0].update(id=""),
            lambda s: s.rows.append(copy.deepcopy(s.rows[0])),
            lambda s: s.header.update(sftdiff=True),
            lambda s: s.header.update(label_alignment="shifted"),
            lambda s: s.header["tokenizer"].update(fingerprint="unknown"),
            lambda s: s.header.update(metadata={"score": float("nan")}),
            lambda s: s.header.update(metadata=[]),
            lambda s: s.header.update(metadata={"packing": True}),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = fixture()
                mutate(data)
                with self.assertRaises(SnapshotError):
                    compare_snapshots(data, fixture())

    def test_broken_json_line_is_actionable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            path.write_text(json.dumps(fixture().header) + '\n{"id":\n', encoding="utf-8")
            with self.assertRaisesRegex(SnapshotError, r":2: invalid JSON"):
                load_snapshot(path)

    def test_rejects_empty_blank_and_nonfinite_json(self):
        for text in ("", "\n", '{"sftdiff":NaN}\n'):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "bad.jsonl"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(SnapshotError):
                    load_snapshot(path)


class ComparisonTests(unittest.TestCase):
    def test_stable_ids_ignore_row_order(self):
        before = fixture()
        other = copy.deepcopy(before.rows[0])
        other["id"] = "b"
        before.rows.append(other)
        after = copy.deepcopy(before)
        after.rows.reverse()
        summary = compare_snapshots(before, after)["summary"]
        self.assertEqual((summary["unchanged"], summary["changed"]), (2, 0))

    def test_label_only_change_and_causal_shift(self):
        before, after = fixture(), fixture()
        after.rows[0]["labels"][0] = 1
        result = compare_snapshots(before, after)
        self.assertEqual(result["rows"][0]["changes"], ["labels"])
        self.assertEqual((result["summary"]["before_targets"], result["summary"]["after_targets"]), (2, 2))

    def test_truncation_and_padding_warnings(self):
        before, after = fixture(), fixture()
        after.rows[0]["labels"] = [-100] * 4
        result = compare_snapshots(before, after)
        self.assertIn("No effective", result["rows"][0]["warnings"]["after"][0])
        after.rows[0]["labels"][-1] = 4
        after.rows[0]["attention_mask"][-1] = 0
        result = compare_snapshots(before, after)
        self.assertIn("Padding", result["rows"][0]["warnings"]["after"][0])

    def test_missing_attention_is_all_ones(self):
        before, after = fixture(), fixture()
        del after.rows[0]["attention_mask"]
        self.assertEqual(compare_snapshots(before, after)["summary"]["changed"], 0)

    def test_missing_records_are_added_and_removed(self):
        before, after = fixture(), fixture()
        after.rows[0]["id"] = "new"
        summary = compare_snapshots(before, after)["summary"]
        self.assertEqual((summary["added"], summary["removed"], summary["changed"]), (1, 1, 0))

    def test_different_vocabulary_is_rejected(self):
        before, after = fixture(), fixture()
        after.header["tokenizer"]["fingerprint"] = "b" * 64
        with self.assertRaisesRegex(SnapshotError, "fingerprints differ"):
            compare_snapshots(before, after)

    def test_comparison_does_not_mutate_inputs(self):
        before, after = fixture(), fixture()
        saved = copy.deepcopy((before, after))
        compare_snapshots(before, after)
        self.assertEqual((before, after), saved)

    def test_inserted_token_has_correct_ranges(self):
        before, after = fixture(), fixture()
        for key, value in [("input_ids", 5), ("labels", -100), ("tokens", "new"), ("attention_mask", 1)]:
            after.rows[0][key].insert(1, value)
        row = compare_snapshots(before, after)["rows"][0]
        self.assertIn(("insert", 1, 1, 1, 2), row["opcodes"])


class CaptureTests(unittest.TestCase):
    def test_capture_exact_arrays_without_retokenizing(self):
        batch = {"input_ids": [[0, 1, 6, 7]], "labels": [[-100, -100, 6, 7]]}
        saved = copy.deepcopy(batch)
        with tempfile.TemporaryDirectory() as folder:
            snapshot = capture_batch(Path(folder) / "a.jsonl", batch, DemoTokenizer(), row_ids=["a"])
        self.assertEqual(snapshot.rows[0]["input_ids"], batch["input_ids"][0])
        self.assertEqual(snapshot.rows[0]["labels"], batch["labels"][0])
        self.assertEqual(batch, saved)

    def test_rejects_missing_or_duplicate_row_ids(self):
        batch = {"input_ids": [[0], [1]], "labels": [[-100], [-100]]}
        for row_ids in (["a"], ["a", "a"], "ab", None):
            with self.subTest(row_ids=row_ids), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(SnapshotError):
                    capture_batch(Path(folder) / "a.jsonl", batch, DemoTokenizer(), row_ids=row_ids)

    def test_optional_real_torch_tensors(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional and not installed")
        batch = {"input_ids": torch.tensor([[0, 1, 6, 7]]), "labels": torch.tensor([[-100, -100, 6, 7]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1]])}
        with tempfile.TemporaryDirectory() as folder:
            output = capture_batch(Path(folder) / "a.jsonl", batch, DemoTokenizer(), row_ids=["a"])
        self.assertEqual(output.rows[0]["labels"], [-100, -100, 6, 7])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 6, 7]])

    def test_bad_batch_fails_before_writing(self):
        cases = [
            {"input_ids": [[0.5]], "labels": [[-100]]},
            {"input_ids": [0], "labels": [-100]},
            {"input_ids": None, "labels": None},
            {"input_ids": [[999]], "labels": [[-100]]},
            {"input_ids": [[0]], "labels": [[-100]], "attention_mask": [[[1]]]},
        ]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "bad.jsonl"
            for batch in cases:
                with self.subTest(batch=batch), self.assertRaises(SnapshotError):
                    capture_batch(target, batch, DemoTokenizer(), row_ids=["a"])
                self.assertFalse(target.exists())


class ReportTests(unittest.TestCase):
    def test_escapes_user_controlled_strings(self):
        before, after = fixture(), fixture()
        malicious = '</script><img src=x onerror="alert(1)">'
        for data in (before, after):
            data.rows[0]["id"] = malicious
            data.rows[0]["tokens"][0] = malicious
            data.header["metadata"] = {"user": malicious}
        report = render_report(compare_snapshots(before, after), before_name=malicious)
        self.assertNotIn(malicious, report)
        self.assertIn("&lt;img", report)
        self.assertEqual(report.count("<script>"), 1)
        self.assertNotIn("<img", report)

    def test_script_hash_and_no_external_assets(self):
        import base64
        report = render_report(compare_snapshots(fixture(), fixture()))
        script = report.split("<script>", 1)[1].split("</script>", 1)[0]
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        self.assertIn(f"sha256-{digest}", report)
        class Assets(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for key, value in attrs:
                    if key in ("src", "href"):
                        raise AssertionError(f"Unexpected external asset: {value}")
        Assets().feed(report)

    def test_html_limit_preserves_full_summary(self):
        before, after = fixture(), fixture()
        after.rows[0]["id"] = "new"
        result = compare_snapshots(before, after)
        report = render_report(result, max_rows=1)
        self.assertIn("Showing 1 of 2 records", report)
        self.assertEqual(report.count('class="record"'), 1)


class CLITests(unittest.TestCase):
    def run_cli(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(args)
        return code, out.getvalue(), err.getvalue()

    def test_demo_and_compare_exit_codes(self):
        with tempfile.TemporaryDirectory() as folder:
            code, out, err = self.run_cli(["demo", "--out", folder])
            self.assertEqual(code, 0, err)
            self.assertIn("4 changed", out)
            self.assertIn("12 -> 14", out)
            self.assertIn("SYNTHETIC DEMO", (Path(folder) / "report.html").read_text(encoding="utf-8"))
            args = ["compare", str(Path(folder) / "before.jsonl"), str(Path(folder) / "after.jsonl")]
            self.assertEqual(self.run_cli(args)[0], 0)
            self.assertEqual(self.run_cli(args + ["--fail-on-change"])[0], 1)
            self.assertEqual(self.run_cli(args + ["--fail-on-warning"])[0], 1)
            self.assertEqual(self.run_cli(["demo", "--out", folder])[0], 2)

    def test_cannot_overwrite_inputs_even_with_force(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.jsonl"
            write_snapshot(path, fixture())
            original = path.read_bytes()
            code, _, err = self.run_cli(["compare", str(path), str(path), "--html", str(path), "--force"])
            self.assertEqual(code, 2)
            self.assertIn("overwrite an input", err)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_file_has_clean_error(self):
        code, _, err = self.run_cli(["compare", "missing-before.jsonl", "missing-after.jsonl"])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
