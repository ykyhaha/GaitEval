# GaitEval

Six native gait tasks evaluated with one scoring protocol. This compact release
includes held-out Summary-SFT predictions and the core code for motion-summary
preparation, local LLM inference, LoRA-SFT and numerical LR / linear-SVM references.

## Recompute the included results

```sh
python3 score.py
```

Python 3.9+; this command needs no GPU, downloads, credentials or third-party
packages. It recomputes the six Summary-SFT rows in main-paper Table 5 and writes
`results.csv`. These are scores from the included predictions, not a new model run.

## Key code

| File | Purpose |
|---|---|
| `gait_io.py` | Motion-summary prompts, native labels, parsing and cohort validation |
| `prepare_sft.py` | Convert the original partitioned summaries into chat examples |
| `train_lora.py` | Qwen2.5-7B LoRA-SFT with completion-only label loss |
| `infer.py` | Generate new predictions through a local model server and score them |
| `linear_reference.py` | Train-imputed and train-scaled LR / linear SVM |
| `score.py` | Recompute the six released score rows |

See **[data formats, dependencies and runnable commands](docs/reproduction.md)**.
The commands accept all six datasets. Actual model runs require the original
processed inputs, their exact partitions and label metadata, and model weights.
Those inputs and trained adapters are not bundled here. The repository does not
yet reproduce every experiment in the paper from raw poses or video.

## Protocol

One prediction per test record; each dataset retains its fixed native class set.
Invalid outputs count as incorrect and as a false negative for their true class;
they are not an additional macro-F1 class. Accuracy includes all records, and
Cohen's kappa uses empirical marginals. Full runs check test IDs and true labels
against the released cohort; different cohorts are not silently substituted.

ASD-Kinect: ASD/typical; ATAXIA: normal/imitated ataxia; CARE-PD: UPDRS gait 0–3;
GMFCS: levels I–IV; PARKINSONISM: low/high motor severity; PATHOGAIT: six performed
gait patterns.

## Checks

```sh
python3 -m unittest discover -s tests -v
```

CPU checks cover data and label handling, prompt redaction, test-cohort validation,
and inference against a local mock endpoint. The mock checks do not measure model
quality. GPU training and generation have not been rerun for this code release.

No manuscript sources, clinical recordings, model checkpoints or paper working
archives are included.

Project website: https://ykyhaha.github.io/GaitEval-v2/

