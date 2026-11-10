# GaitEval

One scoring protocol, six native label spaces. This compact release recomputes the six **summary-SFT** rows in main-paper Table 5 from existing held-out predictions.

## Run

```sh
python3 score.py
```

Python 3.9+; no third-party packages, GPU, downloads or credentials. The command verifies record counts, unique IDs, fixed-label macro-F1 against the paper and writes `results.csv`.

## Protocol

One prediction per test record. Preserve each dataset’s native class set. Invalid outputs count as incorrect and as a false negative for their true class; they are not an additional macro-F1 class. Accuracy includes all records. Cohen’s kappa uses empirical marginals. The table and `datasets.json` identify the exact tested records; data from other experimental cohorts are not substituted.

ASD-Kinect: ASD/typical; ATAXIA: normal/imitated ataxia; CARE-PD: UPDRS gait 0–3; GMFCS: levels I–IV; PARKINSONISM: low/high motor severity; PATHOGAIT: six performed gait patterns.

## Scope

The supplied records contain IDs, true labels and parsed predictions. They reproduce reported scores, not model inference or retraining. Source poses, private clinical recordings and model checkpoints are not redistributed. Additional experiments in the manuscript are outside this deliberately small release. No second scoring protocol, manuscript sources or paper working archives are included.

Project website: https://gaiteval-six-dataset-v2.ky20545083.chatgpt.site/
