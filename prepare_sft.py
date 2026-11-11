"""Prepare motion-summary chat examples from existing, fixed data partitions."""
import argparse
import json
from pathlib import Path

from gait_io import (TASKS, build_prompt, load_meta, read_jsonl, sample_id,
                     validate_test, write_jsonl)


def load_splits(data_dir, dataset, meta):
    data_dir = Path(data_dir)
    records_file = data_dir / "records_slim.jsonl"
    out = {s: [] for s in ("train", "val", "test")}
    if records_file.exists():
        summaries = {}
        for r in read_jsonl(data_dir / "04_motion_summary.jsonl"):
            sid = sample_id(r)
            if sid in summaries:
                raise ValueError(f"Duplicate summary: {sid}")
            summaries[sid] = r.get("text_en") or r.get("summary_text") or ""
        for r in read_jsonl(records_file):
            if dataset == "care_pd" and r.get("label_task") != "UPDRS_GAIT":
                continue
            if r.get("label") in (None, "", "None", "null"):
                continue
            split = r["split"]
            if split not in out:
                raise ValueError(f"Unknown split: {split}")
            out[split].append({**r, "id": sample_id(r),
                               "summary_text": summaries.get(sample_id(r), "")})
    else:
        for split in out:
            prefix = meta.get("input_splits", {}).get(
                split, split + "_sd" if dataset == "gmfcs" else split)
            path = data_dir / f"{prefix}_motion_summary.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            for r in read_jsonl(path):
                if "id" not in r and "sample_id" not in r:
                    r = {**r, "id": f"{prefix}_{r['sample_index']}"}
                out[split].append({**r, "id": sample_id(r),
                                   "summary_text": r.get("summary_text") or r.get("text") or ""})
    return out


def prepare(data_dir, dataset, meta, out_dir):
    splits = load_splits(data_dir, dataset, meta)
    labels = list(meta["label_map"].values())
    seen, subjects, prepared = set(), {}, {}
    for split, rows in splits.items():
        prepared[split] = []
        subjects[split] = set()
        for row in rows:
            sid = sample_id(row)
            if sid in seen:
                raise ValueError(f"Duplicate ID across partitions: {sid}")
            seen.add(sid)
            label = meta["native_map"][str(row["label"])]
            prompt = build_prompt(row["summary_text"], labels, meta["prompt_template"])
            rec = {"id": sid, "label": label, "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": labels[label]}]}
            subject = row.get("subject_id")
            if subject is not None:
                rec["subject_id"] = str(subject)
                subjects[split].add(str(subject))
            prepared[split].append(rec)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if subjects[a] & subjects[b]:
            raise ValueError(f"Subjects overlap between {a} and {b}")
    if not prepared["train"]:
        raise ValueError("Training partition is empty")
    validate_test([{"id": r["id"], "true": r["label"]} for r in prepared["test"]], dataset)
    out_dir = Path(out_dir)
    if any((out_dir / f"{s}.jsonl").exists() for s in prepared):
        raise FileExistsError("Choose an output directory without existing split files")
    for split, rows in prepared.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    exported = {k: v for k, v in meta.items() if k != "native_map"}
    exported.update(dataset=dataset, counts={s: len(v) for s, v in prepared.items()})
    (out_dir / "meta.json").write_text(json.dumps(exported, indent=2) + "\n")
    print(exported["counts"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=TASKS)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--meta", required=True, help="Original label-map / candidate metadata JSON")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    prepare(args.data_dir, args.dataset, load_meta(args.meta, args.dataset), args.out_dir)


if __name__ == "__main__":
    main()
