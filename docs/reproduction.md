# Running the core experiments

The release uses one scoring protocol and the six native task label sets. The
input partitions are supplied, not randomly regenerated. `datasets.json` and
`predictions/` identify the released Summary-SFT evaluation records. Full inference
and reference runs must match those record IDs **and** true labels. A mismatch
raises an error instead of silently scoring a different cohort.

## What is included

| Entry point | Input | Output |
|---|---|---|
| `score.py` | Included predictions | Six recomputed score rows |
| `prepare_sft.py` | Author-prepared summaries, partitions and label metadata | Chat-format train/val/test files |
| `train_lora.py` | Prepared training chats and Qwen2.5-7B weights | Final adapter; optionally merged model |
| `infer.py` | Prepared test chats and a running local model server | New predictions, raw responses and metrics |
| `linear_reference.py` | Author-prepared feature CSV and label metadata | LR or linear-SVM predictions and metrics |

`gait_io.py` contains shared prompt construction, label parsing and cohort checks.
Neither the labels nor reference assistant completions are sent to the model
as test evidence: inference sends only the user turn.

## Data inputs

Obtain the datasets under their original access conditions and use the exact
processed partitions, joint conventions and summary exports used for the paper.
This compact repository does not distribute poses, clinical recordings,
model checkpoints or an end-to-end raw-video preprocessing pipeline. A fresh
split with the same number of samples is not an equivalent input.

Dataset identifiers are `asd_kinect`, `ataxia`, `care_pd`, `gmfcs`,
`parkinsonism` and `pathogait`.

The preparation script accepts either of these existing storage layouts:

- `records_slim.jsonl` plus `04_motion_summary.jsonl`, joined by `sample_id`.
  Records carry `label`, `split`, `subject_id`, and `label_task` for CARE-PD.
  Summary rows carry `sample_id` and `text_en`. CARE-PD is restricted to
  `label_task=UPDRS_GAIT` and its four gait scores.
- Per-split summary files such as `train_motion_summary.jsonl`,
  `val_motion_summary.jsonl`, and `test_motion_summary.jsonl`. Each row carries
  `label`, `summary_text` (or `text`), and `id` / `sample_id` / `sample_index`.
  The GMFCS filenames and generated IDs use the `train_sd`, `val_sd`, `test_sd`
  prefixes already present in its released record IDs. If the exported files
  have other names, specify their exact prefixes under `input_splits` in the
  metadata; explicit IDs should be retained.

Label metadata must be taken from the corresponding processed dataset. It can
contain a zero-based `label_map`, or parallel `labels` (native values) and
`candidates` (prompt-facing class names). Candidate names and their order must
match the original experiment. Do not replace them with newly invented synonyms.
The class count is checked against `datasets.json`; numerical ID / label pairs
are checked against the released test predictions.

Prompt wording and prefix parsing follow the source experiment settings through
`prompt_template` and `prefix_fallback` in the exported metadata. These settings
are also recorded with inference results. They do not change the test partition,
fixed native class set or metric definition. Train and test use the same
prompt-construction function. Subject overlap is rejected when subject IDs are
present; inputs without subject IDs require the original partition metadata.

## Example commands

Use an extracted summary directory and its original label metadata:

```sh
python prepare_sft.py --dataset asd_kinect \
  --data-dir data/asd_kinect --meta data/asd_kinect/meta.json \
  --out-dir prepared/asd_kinect
```

Start a local OpenAI-compatible model server in its own GPU environment. For
example, with vLLM installed there:

```sh
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name qwen7b --port 8000
python infer.py --dataset asd_kinect --data-dir prepared/asd_kinect \
  --model qwen7b --max-tokens 48 --out-dir runs/asd_kinect/zs
```

Select the output-token budget from the corresponding experiment. It is a
required argument: the archived launchers use different budgets. `--limit 2`
provides a partial pipeline smoke test and marks the output as partial; it is
not a full benchmark result. HTTP failures stop the run rather than removing
records from the scoring denominator. Use a new output directory for each run.

For Qwen2.5-7B LoRA-SFT, use a separate training environment:

```sh
python -m pip install -r requirements-train.txt
python train_lora.py --base Qwen/Qwen2.5-7B-Instruct \
  --data-dir prepared/asd_kinect --out-dir models/asd_kinect \
  --epochs 3 --lr 1e-4 --max-seq-len 1024 --bsz 8 --grad-accum 2 --lora-r 16
```

The recipe retains rank 16, alpha 32, dropout 0.05, all linear projections,
completion-only loss, cosine scheduling with warmup 0.03, BF16, and the final
adapter without validation checkpoint selection. Effective batch size 16 assumes
a single training process. Examples exceeding the sequence limit are rejected
so that their label completion cannot be truncated. The source script records
TRL 0.11.4, PEFT 0.13.2 and Transformers 4.57.3; the requirements file preserves
those API versions, but is not a complete lock of the original CUDA environment.

Serve the resulting merged model and evaluate it through the same inference
entry point:

```sh
vllm serve models/asd_kinect/merged --served-model-name gaiteval-sft --port 8000
python infer.py --dataset asd_kinect --data-dir prepared/asd_kinect \
  --model gaiteval-sft --max-tokens 48 --out-dir runs/asd_kinect/sft
```

For the numerical references:

```sh
python -m pip install -r requirements.txt
python linear_reference.py --dataset asd_kinect \
  --features data/asd_kinect/03_features_raw_and_z.csv \
  --meta data/asd_kinect/meta.json --model lr --out-dir runs/asd_kinect/lr
```

The CSV must contain `sample_id`, `subject_id`, `split`, `label`, and `raw__`
feature columns. CARE-PD additionally requires `label_task`. Label values use
the native mapping in the original metadata. Median imputation, scaling and
the selection of observed feature columns are fit on training rows only.
Use `--model svm` for the linear SVM and the experiment's class-weight setting.

The same commands accept all six dataset identifiers. A complete new training
or inference run still requires the original processed inputs and model access.
The included prediction files support immediate score reproduction; they do
not contain the motion summaries or trained adapters needed for these runs.

## Code sources

Core implementations are adapted from
[`james20141606/gaiteval-wacv` at commit
`2348b11f69a248ec57234c92747dda79f368d798`](https://github.com/james20141606/gaiteval-wacv/tree/2348b11f69a248ec57234c92747dda79f368d798).

| Public file | Source implementation |
|---|---|
| `gait_io.py` | `code/view_run.py`, `code/flat_run.py`, `code/v21_llm_runner.py` |
| `prepare_sft.py` | `code/make_sft_data.py`, `code/v21_make_sft_data.py` |
| `infer.py` | `code/view_run.py`, `code/v21_llm_runner.py` |
| `train_lora.py` | `code/sft_train.py` |
| `linear_reference.py` | LR/SVM pipeline in `code/v2_reference_suite.py` |

The portable entry points remove cluster-specific paths and unrelated experiment
branches. They share the released fixed-label scorer and add input validation;
they are not byte-identical copies of every original launcher. Few-shot,
coordinate-view, transfer and additional control experiments are outside this
small code release. Local checks cover prompt fidelity, parsing, input validation,
mock-server inference, CPU reference execution and the six stored score rows.
GPU retraining and model inference have not been rerun as part of this release.
