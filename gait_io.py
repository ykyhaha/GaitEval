"""Shared input, prompt and label handling for the six GaitEval tasks."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = {
    "asd_kinect": ("ASD-Kinect", "compact", False),
    "ataxia": ("ATAXIA", "detailed", True),
    "care_pd": ("CARE-PD", "compact", False),
    "gmfcs": ("GMFCS", "detailed", True),
    "parkinsonism": ("PARKINSONISM", "detailed", True),
    "pathogait": ("PATHOGAIT", "detailed", True),
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_config(dataset):
    return json.loads((ROOT / "datasets.json").read_text())[TASKS[dataset][0]]


def load_meta(path, dataset):
    meta = json.loads(Path(path).read_text())
    if "labels" in meta and "candidates" in meta:
        labels = meta["candidates"]
        values = meta["labels"]
        if len(values) != len(labels) or len(set(map(str, values))) != len(values):
            raise ValueError("Invalid native label mapping")
        native = {str(v): i for i, v in enumerate(values)}
    else:
        lm = {int(k): v for k, v in meta["label_map"].items()}
        if sorted(lm) != list(range(len(lm))):
            raise ValueError("label_map must use consecutive, zero-based class IDs")
        labels = [lm[i] for i in range(len(lm))]
        native = {str(i): i for i in range(len(labels))}
    if (len(labels) != task_config(dataset)["classes"] or
            any(not isinstance(v, str) or not v.strip() for v in labels) or
            len(set(normalize(v) for v in labels)) != len(labels)):
        raise ValueError("Candidate labels do not match this task's fixed class set")
    meta.update(label_map={i: v for i, v in enumerate(labels)}, native_map=native)
    meta.setdefault("prompt_template", TASKS[dataset][1])
    meta.setdefault("prefix_fallback", TASKS[dataset][2])
    if meta["prompt_template"] not in ("compact", "detailed"):
        raise ValueError("Unknown prompt template")
    return meta


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def parse_label(text, labels, prefix_fallback=False):
    """Return the last boundary-anchored label; prefer the longest tied match."""
    text = normalize(text)
    best = None
    for i, label in enumerate(labels):
        name = normalize(label)
        pattern = r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])"
        for match in re.finditer(pattern, text):
            key = (match.start(), len(name), -i)
            if best is None or key > best[0]:
                best = (key, i)
    if best is not None:
        return best[1]
    if prefix_fallback:
        matches = set()
        for i, label in enumerate(labels):
            name = normalize(label)
            for start in range(max(0, len(text) - len(name)), len(text)):
                tail = text[start:]
                if len(tail) >= 3 and name.startswith(tail):
                    matches.add(i)
                    break
        if len(matches) == 1:
            return next(iter(matches))
    return -1


def redact_summary(text):
    text = re.sub(r"(?im)\blabel:\s*[^|\n]+", "label: [REDACTED]", text)
    return re.sub(r"(?im)\bsample_index:\s*\S+", "sample_index: [REDACTED]", text)


def build_prompt(text, labels, template):
    text = redact_summary(text).strip()
    if not text:
        raise ValueError("Motion summary is empty")
    block = "\n".join(f"- {v}" for v in labels)
    if template == "detailed":
        head = (
            "You are classifying a gait clip from a compact biomechanical motion "
            "summary derived from a 3D skeleton sequence. The summary reports motion "
            "amplitudes, left/right asymmetry, trunk/pelvis stability, trajectory "
            "variability, and movement smoothness.\n\n"
        )
    elif template == "compact":
        head = (
            "You are classifying a gait clip from a compact biomechanical motion summary "
            "derived from a skeleton sequence.\n\n"
        )
    else:
        raise ValueError("Unknown prompt template")
    return (head + text + "\n\n"
            "Task: based ONLY on the motion summary above, classify the gait into exactly "
            "one of the following classes:\n" + block + "\n\n"
            "Return only one label, written exactly as listed above.")


def sample_id(row):
    value = row.get("id", row.get("sample_id"))
    if value is None or not str(value).strip():
        raise ValueError("Each record needs an id or sample_id")
    return str(value)


def validate_test(rows, dataset, partial=False):
    """Check both sample membership and ground truth against the released cohort."""
    expected = {r["id"]: r["true"] for r in
                read_jsonl(ROOT / task_config(dataset)["file"])}
    ids = [r["id"] for r in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Test inputs are empty or contain duplicate IDs")
    if not partial and set(ids) != set(expected):
        raise ValueError("Test IDs differ from the released cohort; use its exact partition")
    for row in rows:
        if row["id"] not in expected or row["true"] != expected[row["id"]]:
            raise ValueError(f"Test ID or label mismatch: {row['id']}")
