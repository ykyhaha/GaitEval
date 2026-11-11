"""Fit train-imputed, train-scaled LR or linear SVM on named numerical features."""
import argparse
import csv
import json
import math
from pathlib import Path

from gait_io import TASKS, load_meta, validate_test, write_jsonl
from score import metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=TASKS)
    ap.add_argument("--features", required=True, help="CSV with raw__ feature columns")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", choices=["lr", "svm"], default="lr")
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    meta = load_meta(args.meta, args.dataset)
    with open(args.features, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        cols = [c for c in fields if c.startswith("raw__")]
        if not cols or not {"sample_id", "subject_id", "split", "label"}.issubset(fields):
            raise ValueError("Need sample_id, subject_id, split, label and raw__ features")
        rows = [r for r in reader if r["split"] in ("train", "test")]
    if args.dataset == "care_pd":
        if "label_task" not in fields:
            raise ValueError("CARE-PD feature rows must identify label_task=UPDRS_GAIT")
        rows = [r for r in rows if r["label_task"] == "UPDRS_GAIT"]
    rows = [r for r in rows if r["label"] not in ("", "None", "null", "nan")]
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate feature record IDs")
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    if not train or not test:
        raise ValueError("Need nonempty train and test partitions")
    if {r["subject_id"] for r in train} & {r["subject_id"] for r in test}:
        raise ValueError("Subjects overlap between train and test")
    y = lambda r: meta["native_map"][str(r["label"])]
    checked = [{"id": r["sample_id"], "true": y(r)} for r in test]
    validate_test(checked, args.dataset)

    def matrix(part):
        return np.array([[float(r[c]) if r[c] not in ("", "None", "null", "nan")
                          and math.isfinite(float(r[c])) else np.nan for c in cols]
                         for r in part])

    xtr, xte = matrix(train), matrix(test)
    keep = ~np.all(np.isnan(xtr), axis=0)
    if not keep.any():
        raise ValueError("No observed training features")
    weight = None if args.class_weight == "none" else "balanced"
    model = (LogisticRegression(max_iter=2000, class_weight=weight, random_state=0)
             if args.model == "lr" else
             LinearSVC(max_iter=5000, class_weight=weight, random_state=0))
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    pipe.fit(xtr[:, keep], [y(r) for r in train])
    predictions = pipe.predict(xte[:, keep])
    results = [{**r, "pred": int(p)} for r, p in zip(checked, predictions)]
    out = Path(args.out_dir)
    if (out / "results.jsonl").exists():
        raise FileExistsError("Choose a new output directory")
    write_jsonl(out / "results.jsonl", results)
    summary = {"dataset": args.dataset, "model": args.model,
               "class_weight": args.class_weight, "n": len(test),
               "features": [c for c, k in zip(cols, keep) if k],
               **metrics(results, len(meta["label_map"]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
