import contextlib
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_io import (ROOT, TASKS, build_prompt, load_meta, parse_label, read_jsonl,
                     task_config, validate_test, write_jsonl)
from prepare_sft import prepare
from score import metrics


class PipelineTests(unittest.TestCase):
    def test_boundary_parsing_and_invalid_denominator(self):
        labels = ["GMFCS I", "GMFCS II", "GMFCS III", "GMFCS IV"]
        self.assertEqual(parse_label("FINAL: GMFCS III", labels), 2)
        self.assertEqual(parse_label("not GMFCS II; FINAL: GMFCS IV", labels), 3)
        self.assertEqual(parse_label("abnormal gait", ["normal gait", "ataxic gait"]), -1)
        self.assertEqual(parse_label("ataxic ga", ["normal gait", "ataxic gait"], True), 1)
        m = metrics([{"true": 0, "pred": 0}, {"true": 1, "pred": -1}], 2)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["macro_f1"], 0.5)
        self.assertEqual(m["invalid"], 1)

    def test_ground_truth_redaction(self):
        text = build_prompt("label: SECRET | sample_index: 17 | range=0.4",
                            ["class a", "class b"], "detailed")
        self.assertNotIn("SECRET", text)
        self.assertNotIn("sample_index: 17", text)
        self.assertIn("range=0.4", text)

    def test_exact_membership_and_labels(self):
        for ds in TASKS:
            rows = read_jsonl(ROOT / task_config(ds)["file"])
            validate_test(rows, ds)
            with self.assertRaises(ValueError):
                validate_test(rows[:-1], ds)
            bad = [dict(r) for r in rows]
            bad[0]["true"] = (bad[0]["true"] + 1) % task_config(ds)["classes"]
            with self.assertRaises(ValueError):
                validate_test(bad, ds)

    def test_prepare_and_infer_all_six(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(payload)
                raw = json.dumps({"choices": [{"message": {"content": "class_0"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for ds in TASKS:
                    k = task_config(ds)["classes"]
                    source, dest = root / ds / "source", root / ds / "prepared"
                    names = [f"class_{i}" for i in range(k)]
                    native = [str(i) for i in range(k)]
                    meta_path = root / ds / "meta.json"
                    meta_path.parent.mkdir(parents=True)
                    meta_path.write_text(json.dumps({"labels": native, "candidates": names}))
                    stored = read_jsonl(ROOT / task_config(ds)["file"])
                    records = [{"sample_id": r["id"], "label": str(r["true"]),
                                "subject_id": f"test-person-{i}", "split": "test",
                                "label_task": "UPDRS_GAIT"}
                               for i, r in enumerate(stored)]
                    records += [{"sample_id": f"train-fixture-{i}", "label": str(i),
                                 "subject_id": f"train-person-{i}", "split": "train",
                                 "label_task": "UPDRS_GAIT"} for i in range(k)]
                    # Synthetic summaries exercise the pipeline, not model accuracy.
                    write_jsonl(source / "records_slim.jsonl", records)
                    write_jsonl(source / "04_motion_summary.jsonl", [
                        {"sample_id": r["sample_id"], "text_en": "range=0.4; stability=0.2"}
                        for r in records])
                    with contextlib.redirect_stdout(io.StringIO()):
                        prepare(source, ds, load_meta(meta_path, ds), dest)
                    meta = load_meta(dest / "meta.json", ds)
                    self.assertEqual(meta["native_map"], {str(i): i for i in range(k)})
                    cmd = [sys.executable, str(ROOT / "infer.py"), "--dataset", ds,
                           "--data-dir", str(dest), "--model", "test-fixture",
                           "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                           "--max-tokens", "48", "--limit", "2", "--max-workers", "1",
                           "--out-dir", str(root / ds / "inference")]
                    run = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    summary = json.loads(run.stdout)
                    self.assertEqual(summary["n"], 2)
                    self.assertTrue(summary["partial"])
                self.assertEqual(len(received), 12)
                self.assertTrue(all([m["role"] for m in r["messages"]] == ["user"]
                                    for r in received))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_subject_overlap_and_missing_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "meta.json"
            p.write_text(json.dumps({"label_map": {"0": "ASD", "1": "typical"}}))
            stored = read_jsonl(ROOT / task_config("asd_kinect")["file"])
            records = [{"sample_id": r["id"], "label": str(r["true"]),
                        "split": "test", "subject_id": "same"} for r in stored]
            records += [{"sample_id": "training", "label": "0", "split": "train",
                         "subject_id": "same"}]
            write_jsonl(root / "records_slim.jsonl", records)
            summaries = [{"sample_id": r["sample_id"], "text_en": "range=0.4"} for r in records]
            write_jsonl(root / "04_motion_summary.jsonl", summaries)
            with self.assertRaisesRegex(ValueError, "Subjects overlap"):
                prepare(root, "asd_kinect", load_meta(p, "asd_kinect"), root / "out")
            write_jsonl(root / "04_motion_summary.jsonl", [])
            with self.assertRaisesRegex(ValueError, "empty"):
                prepare(root, "asd_kinect", load_meta(p, "asd_kinect"), root / "out")

    def test_partitioned_summary_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ds in TASKS:
                k = task_config(ds)["classes"]
                folder = root / ds
                folder.mkdir()
                p = folder / "meta.json"
                p.write_text(json.dumps({"label_map": {str(i): f"class_{i}" for i in range(k)}}))
                for split in ("train", "val", "test"):
                    prefix = split + "_sd" if ds == "gmfcs" else split
                    rows = ([{"id": r["id"], "label": r["true"], "summary_text": "range=0.4"}
                             for r in read_jsonl(ROOT / task_config(ds)["file"])]
                            if split == "test" else
                            [{"sample_index": i, "label": i, "summary_text": "range=0.4"}
                             for i in range(k)])
                    write_jsonl(folder / f"{prefix}_motion_summary.jsonl", rows)
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare(folder, ds, load_meta(p, ds), folder / "prepared")
                self.assertEqual(len(read_jsonl(folder / "prepared/test.jsonl")),
                                 task_config(ds)["n"])

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("sklearn"),
                         "Install requirements.txt for CPU reference checks")
    def test_linear_reference_all_six(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ds in TASKS:
                root = Path(tmp) / ds
                root.mkdir()
                k = task_config(ds)["classes"]
                (root / "meta.json").write_text(json.dumps({
                    "labels": [str(i) for i in range(k)],
                    "candidates": [f"class_{i}" for i in range(k)]}))
                # Numerical fixtures test the CLI, not the paper's feature values.
                rows = [{"sample_id": r["id"], "subject_id": f"test-{i}",
                         "label": str(r["true"]), "split": "test", "label_task": "UPDRS_GAIT",
                         "raw__x": str(r["true"]), "raw__unsupported": ""}
                        for i, r in enumerate(read_jsonl(ROOT / task_config(ds)["file"]))]
                rows += [{"sample_id": f"train-{i}", "subject_id": f"train-{i}",
                          "label": str(i % k), "split": "train", "label_task": "UPDRS_GAIT",
                          "raw__x": str(i % k), "raw__unsupported": ""} for i in range(4*k)]
                with (root / "features.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                for model in ("lr", "svm"):
                    cmd = [sys.executable, str(ROOT / "linear_reference.py"), "--dataset", ds,
                           "--features", str(root / "features.csv"), "--meta", str(root / "meta.json"),
                           "--model", model, "--out-dir", str(root / model)]
                    run = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    result = json.loads(run.stdout)
                    self.assertEqual(result["n"], task_config(ds)["n"])
                    self.assertEqual(result["features"], ["raw__x"])


if __name__ == "__main__":
    unittest.main()
