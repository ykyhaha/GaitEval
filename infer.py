"""Run motion-summary classification against a local OpenAI-compatible model server."""
import argparse
import concurrent.futures
import json
from pathlib import Path
from urllib.request import Request, urlopen

from gait_io import (TASKS, load_meta, parse_label, read_jsonl, sample_id,
                     validate_test, write_jsonl)
from score import metrics


def make_inputs(path, labels):
    out = []
    for r in read_jsonl(path):
        messages = r["messages"]
        if (len(messages) != 2 or messages[0]["role"] != "user" or
                messages[1]["role"] != "assistant" or
                not messages[0]["content"].strip()):
            raise ValueError("Expected one user prompt and one reference assistant label")
        y = labels.index(messages[1]["content"])
        if "label" in r and int(r["label"]) != y:
            raise ValueError("Reference completion and numeric label disagree")
        out.append({"id": sample_id(r), "true": y, "messages": [messages[0]],
                    **({"subject_id": str(r["subject_id"])} if "subject_id" in r else {})})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=TASKS)
    ap.add_argument("--data-dir", required=True, help="SFT export with test.jsonl and meta.json")
    ap.add_argument("--model", required=True, help="Exact name served by the local model server")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="Partial smoke test; not a full-cohort paper result")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.max_tokens < 1 or args.max_workers < 1 or (args.limit is not None and args.limit < 1):
        ap.error("Token, worker and limit values must be positive")
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    meta = load_meta(data_dir / "meta.json", args.dataset)
    labels = list(meta["label_map"].values())
    rows = make_inputs(data_dir / "test.jsonl", labels)
    if args.limit is not None:
        rows = rows[:args.limit]
    validate_test(rows, args.dataset, partial=args.limit is not None)
    if (out_dir / "results.jsonl").exists():
        raise FileExistsError("Choose a new output directory for each model run")

    def classify(row):
        payload = json.dumps({"model": args.model, "messages": row["messages"],
                              "temperature": 0, "max_tokens": args.max_tokens}).encode()
        req = Request(args.base_url.rstrip("/") + "/chat/completions", payload,
                      {"Content-Type": "application/json"})
        with urlopen(req, timeout=300) as response:
            raw = json.load(response)["choices"][0]["message"]["content"] or ""
        result = {k: v for k, v in row.items() if k != "messages"}
        result.update(pred=parse_label(raw, labels, meta["prefix_fallback"]), raw_output=raw)
        return result

    with concurrent.futures.ThreadPoolExecutor(args.max_workers) as ex:
        results = list(ex.map(classify, rows))
    summary = {"dataset": args.dataset, "model": args.model, "n": len(results),
               "partial": args.limit is not None, "max_tokens": args.max_tokens,
               "temperature": 0, "prompt_template": meta["prompt_template"],
               "prefix_fallback": meta["prefix_fallback"],
               "label_map": meta["label_map"], **metrics(results, len(labels))}
    write_jsonl(out_dir / "results.jsonl", results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
