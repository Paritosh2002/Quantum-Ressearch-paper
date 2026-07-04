# Quantum ML on GSE84507 — CML Single-Cell qPCR

Benchmarked feasibility study applying **quantum machine learning** (variational
classifiers + a quantum-kernel SVM) against classical baselines on the **GSE84507**
single-cell qPCR dataset (Warfvinge et al., 2017 — chronic myeloid leukemia stem
cells, Fluidigm 96.96, ~96-gene panel, 2,333 cells).

Everything is driven by one CLI (`cli.py`) over a small, extensible package
(`qmlcml/`). Adding a new algorithm is a single file with an `@register` decorator.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

## Commands

```bash
python cli.py preprocess          # download data + series matrix, build results/task_*.npz
python cli.py list-models         # show available algorithms
python cli.py train               # train ALL models on ALL tasks
python cli.py train --model vqc                       # one model, all tasks
python cli.py train --model vqc qsvm --task diagnosis_binary
python cli.py evaluate            # comparison table + results/benchmark_summary.md
python cli.py visualize           # render every figure into figures/
python cli.py all                 # preprocess -> train -> evaluate -> visualize
```

`train` optionally takes `--qsvm-max-train N` (cap QSVM training cells per fold;
`0` = use all — the statevector kernel is fast enough to not need a cap).

## Models (`python cli.py list-models`)

| Representation | Name | Notes |
|---|---|---|
| classical | `svm_rbf` | RBF SVM, `class_weight=balanced` |
| classical | `random_forest` | 300 trees, balanced |
| classical | `logistic_regression` | balanced |
| quantum | `vqc` | AngleEmbedding(RY) + StronglyEntanglingLayers(3), Torch/PennyLane |
| quantum | `qnn_reupload` | data re-uploading + BasicEntanglerLayers (distinct ansatz) |
| quantum | `qsvm` | ZZ feature map + **statevector fidelity kernel** + SVC |

Classical models use the 77-gene z-scored features; quantum models use 8 PCA
components angle-encoded to `[0, π]` (one qubit each).

### Adding an algorithm
Create `qmlcml/models/mymodel.py`, subclass `BaseModel` (or `VariationalModel` for
a trainable circuit), set `name`/`representation`, implement `fit`/`predict`/
`predict_score`, decorate with `@register`, and import it in
`qmlcml/models/__init__.py`. It is then trainable via `--model mymodel`.

## Visualizations (each a separate file in `figures/`)

| File | What it shows |
|---|---|
| `training_curves.png` | per-epoch **train loss** and **train-vs-test accuracy** for the iterative quantum models (mean over folds) |
| `accuracy_comparison.png` | grouped bar chart of CV accuracy, every model × every task |
| `metrics_heatmap.png` | model × metric grid per task |
| `confusion_matrices.png` | pooled out-of-fold confusion matrices |
| `roc_curves.png` | pooled out-of-fold ROC curves |

## Evaluation protocol
Every model shares the **same** `StratifiedKFold(5, shuffle, seed=42)` split.
Quantum PCA + angle encoding is re-fit **inside each fold** on that fold's training
data (no leakage). Metrics: accuracy, AUC-ROC (macro-OVR for 3-class), macro-F1,
macro precision/recall. Out-of-fold predictions and per-epoch histories are stored
in `results/metrics/*.json` and drive the figures.

## The three tasks
| Task | Classes | Cells (post-QC) |
|---|---|---|
| `diagnosis_binary` (primary)   | nBM (0) vs Dx (1)  | 173 vs 1168 |
| `therapy_binary` (secondary)   | Dx (0) vs TKI (1)  | 1168 vs 781 |
| `three_class` (tertiary)       | nBM / Dx / TKI     | 173 / 1168 / 781 |

## Preprocessing — what the real data forced us to change

The implementation brief made several assumptions the actual GEO files contradict.
Verified facts and the resulting decisions (all in `qmlcml/preprocess.py`):

| Brief assumption | Reality in GSE84507 | What the code does |
|---|---|---|
| Sentinel is `999`/`0` | Failed reactions are `40.0` (assay ceiling); 58.6% of readings | Auto-detects the ceiling; `Ct>=40` ⇒ undetected |
| `LoD = 24` | Using 24 silently zeros out 3,515 real low-expression readings | `LoD = 40`, so `expression = 40 − Ct`; "expressed" ⇔ "detected" |
| Labels in sample-name tokens | Column names carry **no** nBM/healthy label | Labels from the **GEO series matrix** metadata, matched by cell title |
| Per-cell QC: drop if >50% zeros | Median dropout ~58% → 0.50 keeps only 370 cells, destroys nBM (177→7) | Threshold relaxed to **0.70** (2,122 cells, class balance intact) |
| nBM roughly balanced | nBM is a strong minority (173 vs 1168) | `class_weight="balanced"` + class-weighted VQC loss |

Pipeline: download data **and** series matrix → transpose (genes-in-rows →
cells×genes) → map labels by title, drop noRT/unknown → mask `Ct>=40` →
`expr = 40−Ct` → per-cell QC (`>70%` dropout) → per-gene QC (`<10%` detected) →
per task: stratified 80/20 split, `StandardScaler` + `PCA(8)` + angle-scale to
`[0, π]`, **all fit on train only**.

### On the quantum kernel
Qiskit's `FidelityQuantumKernel` runs one circuit per pair — O(n²) circuit
executions, ~2 min for an 80×80 kernel on CPU. Since we simulate exactly, `qsvm`
instead computes each sample's embedded **statevector once** and forms the Gram
matrix as `|S Sᴴ|²`. This is numerically identical to the fidelity kernel
(verified to 1e-12) but runs in seconds, so no subsampling is required.

## Layout
```
cli.py                     unified command-line entry point
qmlcml/
  config.py                paths, dataset + CV constants
  preprocess.py            download + clean + build task .npz files
  data.py                  task loading, per-fold quantum encoding, CV splits
  metrics.py               metric computation + result serialization
  train.py                 generic CV runner (any registered model)
  report.py                console table + markdown summary
  visualize.py             all figures
  models/
    base.py                BaseModel + registry
    classical.py           svm_rbf / random_forest / logistic_regression
    variational.py         vqc / qnn_reupload (+ shared training loop)
    qsvm.py                quantum-kernel SVM (statevector fidelity)
results/                   task_*.npz, metrics/*.json, benchmark_summary.md
figures/                   the five PNGs above
```
