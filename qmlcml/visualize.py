"""All plotting. Each function writes a separate PNG under figures/.

  training_curves.png      train loss / train acc / val acc vs epoch (iterative models)
  accuracy_comparison.png  grouped bar chart of CV accuracy per model per task
  metrics_heatmap.png      model x metric grid per task
  confusion_matrices.png   pooled out-of-fold confusion matrices
  roc_curves.png           pooled out-of-fold ROC curves
"""

import os
import numpy as np

from . import config as C
from .metrics import load_results


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, name):
    os.makedirs(C.FIGURES_DIR, exist_ok=True)
    path = os.path.join(C.FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  figure -> {os.path.relpath(path, C.ROOT)}")
    return path


def _mean_history(hist_list, key):
    """Average a per-fold history list to (epochs,) with equal-length trimming."""
    series = [h[key] for h in hist_list if key in h and h[key]]
    if not series:
        return None
    n = min(len(s) for s in series)
    return np.mean([s[:n] for s in series], axis=0)


# --- training/testing curves -------------------------------------------------
def training_curves(rows=None):
    plt = _mpl()
    rows = rows or load_results()
    iters = [r for r in rows if r.get("history")]
    if not iters:
        print("  (no iterative-model histories found; skipping training_curves)")
        return
    models = sorted({r["model"] for r in iters})
    fig, axes = plt.subplots(2, len(C.TASKS),
                             figsize=(4.6 * len(C.TASKS), 7.5), squeeze=False)
    for ti, task in enumerate(C.TASKS):
        ax_loss, ax_acc = axes[0][ti], axes[1][ti]
        for model in models:
            r = next((r for r in iters if r["task"] == task and r["model"] == model), None)
            if not r:
                continue
            tl = _mean_history(r["history"], "train_loss")
            ta = _mean_history(r["history"], "train_acc")
            va = _mean_history(r["history"], "val_acc")
            ep = np.arange(1, len(tl) + 1) if tl is not None else None
            if tl is not None:
                ax_loss.plot(ep, tl, label=model)
            if ta is not None:
                ax_acc.plot(ep, ta, label=f"{model} train")
            if va is not None:
                ax_acc.plot(ep, va, "--", label=f"{model} test")
        ax_loss.set_title(C.TASK_TITLES.get(task, task), fontsize=10)
        ax_loss.set_ylabel("train loss"); ax_loss.set_xlabel("epoch")
        ax_loss.legend(fontsize=7)
        ax_acc.set_ylabel("accuracy"); ax_acc.set_xlabel("epoch")
        ax_acc.set_ylim(0, 1); ax_acc.legend(fontsize=7)
    fig.suptitle("Training dynamics (mean over 5 folds): loss (top), "
                 "train vs test accuracy (bottom)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "training_curves.png")


# --- accuracy comparison (its own file) --------------------------------------
def accuracy_comparison(rows=None):
    plt = _mpl()
    rows = rows or load_results()
    if not rows:
        print("  (no results; skipping accuracy_comparison)")
        return
    models = sorted({r["model"] for r in rows})
    x = np.arange(len(C.TASKS))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(2.2 * len(C.TASKS) + 3, 5))
    for mi, model in enumerate(models):
        accs, errs = [], []
        for task in C.TASKS:
            r = next((r for r in rows if r["task"] == task and r["model"] == model), None)
            accs.append(r["mean"]["accuracy"] if r else 0)
            errs.append(r["std"]["accuracy"] if r else 0)
        bars = ax.bar(x + mi * width, accs, width, yerr=errs, capsize=3, label=model)
        for b, a in zip(bars, accs):
            if a:
                ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([C.TASK_TITLES.get(t, t) for t in C.TASKS], fontsize=9)
    ax.set_ylabel("5-fold CV accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy comparison: classical vs quantum models")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "accuracy_comparison.png")


# --- metric heatmap ----------------------------------------------------------
def metrics_heatmap(rows=None):
    plt = _mpl()
    rows = rows or load_results()
    if not rows:
        return
    metrics = ["accuracy", "auc", "f1_macro", "precision_macro", "recall_macro"]
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(C.TASKS), figsize=(4.5 * len(C.TASKS), 4), squeeze=False)
    for ti, task in enumerate(C.TASKS):
        ax = axes[0][ti]
        grid = np.full((len(models), len(metrics)), np.nan)
        for mi, model in enumerate(models):
            r = next((r for r in rows if r["task"] == task and r["model"] == model), None)
            if r:
                for ci, met in enumerate(metrics):
                    grid[mi, ci] = r["mean"].get(met, np.nan)
        im = ax.imshow(grid, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
        for (a, b), v in np.ndenumerate(grid):
            if v == v:
                ax.text(b, a, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.75 else "black", fontsize=7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace("_macro", "") for m in metrics], rotation=30, fontsize=8)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
        ax.set_title(C.TASK_TITLES.get(task, task), fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="score")
    fig.suptitle("Metric heatmap (mean over 5 folds)", fontsize=11)
    _save(fig, "metrics_heatmap.png")


# --- confusion matrices ------------------------------------------------------
def confusion_matrices(rows=None):
    from sklearn.metrics import confusion_matrix
    plt = _mpl()
    rows = [r for r in (rows or load_results()) if "oof" in r]
    if not rows:
        print("  (no OOF predictions; skipping confusion_matrices)")
        return
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(len(C.TASKS), len(models),
                             figsize=(2.9 * len(models), 2.9 * len(C.TASKS)), squeeze=False)
    for ti, task in enumerate(C.TASKS):
        names = C.CLASS_NAMES.get(task)
        for mi, model in enumerate(models):
            ax = axes[ti][mi]
            r = next((r for r in rows if r["task"] == task and r["model"] == model), None)
            if not r:
                ax.axis("off"); continue
            yt, yp = np.array(r["oof"]["y_true"]), np.array(r["oof"]["y_pred"])
            cm = confusion_matrix(yt, yp)
            ax.imshow(cm, cmap="Blues")
            for (a, b), v in np.ndenumerate(cm):
                ax.text(b, a, int(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black", fontsize=8)
            n = cm.shape[0]
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(names, fontsize=7); ax.set_yticklabels(names, fontsize=7)
            if mi == 0: ax.set_ylabel(f"{task}\ntrue", fontsize=8)
            if ti == 0: ax.set_title(model, fontsize=9)
            if ti == len(C.TASKS) - 1: ax.set_xlabel("predicted", fontsize=8)
    fig.suptitle("Confusion matrices (pooled out-of-fold predictions)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "confusion_matrices.png")


# --- ROC curves --------------------------------------------------------------
def roc_curves(rows=None):
    from sklearn.metrics import roc_curve, auc as sk_auc
    from sklearn.preprocessing import label_binarize
    plt = _mpl()
    rows = [r for r in (rows or load_results()) if "oof" in r]
    if not rows:
        print("  (no OOF predictions; skipping roc_curves)")
        return
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(C.TASKS), figsize=(5 * len(C.TASKS), 4.5), squeeze=False)
    for ti, task in enumerate(C.TASKS):
        ax = axes[0][ti]
        for model in models:
            r = next((r for r in rows if r["task"] == task and r["model"] == model), None)
            if not r:
                continue
            yt = np.array(r["oof"]["y_true"]); ys = np.array(r["oof"]["y_score"])
            n = len(np.unique(yt))
            if n == 2:
                fpr, tpr, _ = roc_curve(yt, ys)
                ax.plot(fpr, tpr, label=f"{model} (AUC={sk_auc(fpr, tpr):.2f})")
            else:
                yb = label_binarize(yt, classes=sorted(np.unique(yt)))
                fpr, tpr, _ = roc_curve(yb.ravel(), ys.ravel())
                ax.plot(fpr, tpr, label=f"{model} (micro AUC={sk_auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(C.TASK_TITLES.get(task, task), fontsize=10)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("ROC curves (pooled out-of-fold predictions)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "roc_curves.png")


def all_figures():
    rows = load_results()
    if not rows:
        print("No results found. Train some models first.")
        return
    print("Rendering figures:")
    training_curves(rows)
    accuracy_comparison(rows)
    metrics_heatmap(rows)
    confusion_matrices(rows)
    roc_curves(rows)
