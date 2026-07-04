"""Console + markdown comparison tables from stored metrics."""

import os
from . import config as C
from .metrics import load_results


def print_table(rows=None):
    rows = rows or load_results()
    if not rows:
        print("No metrics found. Train some models first.")
        return
    hdr = f"{'Model':<22}{'Acc':>8}{'AUC':>8}{'F1':>8}{'Prec':>8}{'Rec':>8}"
    for task in C.TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        print(f"\n{C.TASK_TITLES.get(task, task)}")
        print(hdr); print("-" * len(hdr))
        for r in sorted(task_rows, key=lambda r: -r["mean"]["accuracy"]):
            m = r["mean"]
            print(f"{r['model']:<22}{m['accuracy']:>8.3f}{m['auc']:>8.3f}"
                  f"{m['f1_macro']:>8.3f}{m['precision_macro']:>8.3f}{m['recall_macro']:>8.3f}")


def write_markdown(rows=None, path=None):
    rows = rows or load_results()
    if not rows:
        return
    path = path or os.path.join(C.RESULTS_DIR, "benchmark_summary.md")
    lines = ["# GSE84507 QML Benchmark Summary\n",
             "Mean over 5-fold stratified CV (std in parentheses). "
             "Classical models use 77 z-scored genes; quantum models use 8 PCA "
             "components angle-encoded to [0, pi].\n"]
    for task in C.TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        lines += [f"\n## {C.TASK_TITLES.get(task, task)}\n",
                  "| Model | Repr. | Accuracy | AUC | Macro-F1 | Precision | Recall |",
                  "|---|---|---|---|---|---|---|"]
        for r in sorted(task_rows, key=lambda r: -r["mean"]["accuracy"]):
            m, s = r["mean"], r["std"]
            lines.append(
                f"| {r['model']} | {r.get('representation','?')} "
                f"| {m['accuracy']:.3f} ({s['accuracy']:.3f}) "
                f"| {m['auc']:.3f} ({s['auc']:.3f}) "
                f"| {m['f1_macro']:.3f} ({s['f1_macro']:.3f}) "
                f"| {m['precision_macro']:.3f} | {m['recall_macro']:.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nMarkdown summary -> {os.path.relpath(path, C.ROOT)}")
