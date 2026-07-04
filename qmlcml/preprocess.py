"""Preprocessing pipeline for GSE84507 (single-cell qPCR, CML stem cells).

Key facts established by inspecting the real files (NOT assumed from the brief):
  * Raw table is GENES-IN-ROWS x CELLS-IN-COLUMNS (95 x 2333) -> transpose.
  * Values are Ct. Failed-reaction sentinel is 40.0 (assay ceiling); 58.6% of
    readings are 40 -> undetected. So LoD = 40 and expression = LoD - Ct, making
    "expressed" (expr>0) equivalent to "detected" (Ct<40).
  * Column names carry NO healthy/nBM label. Authoritative per-cell labels come
    from the GEO series matrix (disease state + clinical phase), matched by title:
        healthy control (nBM) -> 0   [180 cells]
        at diagnosis (Dx)     -> 1   [1264 cells]
        after TKI treatment   -> 2   [889 cells]
  * Per-cell QC threshold relaxed to 0.70 (this data's median dropout is ~58%;
    the brief's 0.50 would keep only 370 cells and destroy the minority nBM class).

Outputs results/task_{diagnosis_binary,therapy_binary,three_class}.npz.
"""

import os
import gzip
import urllib.request

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from . import config as C


def _download(url, path):
    if os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} present ({os.path.getsize(path)} bytes)")
        return
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, path)
    print(f"  saved {os.path.basename(path)} ({os.path.getsize(path)} bytes)")


def download_data():
    print("Downloading inputs ...")
    _download(C.DATA_URL, C.LOCAL_DATA)
    _download(C.MATRIX_URL, C.LOCAL_MATRIX)


def load_expression_table(path):
    raw = pd.read_csv(path, sep="\t", index_col=0)
    print("\n=== INSPECTION REPORT -- READ BEFORE PROCEEDING ===")
    print(f"Shape as loaded (rows x cols): {raw.shape}")
    print(f"First 3 row names:  {raw.index[:3].tolist()}")
    print(f"First 3 col names:  {raw.columns[:3].tolist()}")
    vals = pd.to_numeric(pd.Series(raw.values.flatten()), errors="coerce").dropna()
    ceiling = vals.max()
    print(f"Value range: {vals.min()} .. {ceiling}")
    print(f"Detected sentinel/ceiling: {ceiling} "
          f"({(vals == ceiling).mean()*100:.1f}% of readings == ceiling -> failed)")
    if abs(ceiling - C.LOD) > 1e-6:
        print(f"  !! WARNING: detected ceiling {ceiling} != configured LOD {C.LOD}")
    else:
        print(f"  OK: configured LOD ({C.LOD}) matches the detected ceiling.")
    print("=== END REPORT ===\n")

    first_row = str(raw.index[0])
    if any(ch.isalpha() for ch in first_row) and len(first_row) < 30:
        print("Detected genes-in-rows -> transposing to (cells x genes)")
        raw = raw.T
    else:
        print("Detected cells-in-rows -> no transpose needed")
    return raw


def _split_matrix_line(line):
    return [p.strip().strip('"') for p in line.rstrip("\n").split("\t")[1:]]


def load_labels_from_matrix(path):
    titles = disease = phase = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                titles = _split_matrix_line(line)
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = _split_matrix_line(line)
                joined = " ".join(vals).lower()
                if "disease state" in joined:
                    disease = vals
                elif "clinical phase" in joined:
                    phase = vals
    if titles is None or disease is None or phase is None:
        raise RuntimeError("Could not locate title/disease/clinical-phase rows in matrix")

    labels = {}
    for t, d, p in zip(titles, disease, phase):
        dl, pl = d.lower(), p.lower()
        if "healthy control" in dl or "nbm" in dl:
            labels[t] = "nBM"
        elif "tki" in pl:
            labels[t] = "TKI"
        elif "diagnosis" in pl or "(dx)" in pl:
            labels[t] = "Dx"
        else:
            labels[t] = "unknown"
    return pd.Series(labels)


def attach_labels(expr, title_to_label):
    raw_labels = pd.Series(
        [title_to_label.get(idx, "unmatched") for idx in expr.index], index=expr.index)
    is_nort = expr.index.to_series().str.contains("nort", case=False, regex=False)
    raw_labels[is_nort.values] = "noRT"
    n_unmatched = int((raw_labels == "unmatched").sum())
    if n_unmatched:
        print(f"  note: {n_unmatched} cells had no matching metadata title (dropped)")
    keep = ~raw_labels.isin(["noRT", "unknown", "unmatched"])
    expr, labels = expr[keep.values], raw_labels[keep.values]
    print(f"After dropping noRT/unknown/unmatched: {expr.shape[0]} cells")
    print("Class distribution (pre-QC):")
    print(labels.value_counts().to_string(), "\n")
    return expr, labels


def convert_ct_to_expression(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df[df >= C.LOD] = C.LOD
    expr = C.LOD - df
    expr[expr < 0] = 0
    return expr.fillna(0)


def qc_filter(expr, labels):
    zero_frac = (expr == 0).sum(axis=1) / expr.shape[1]
    keep_cells = zero_frac <= C.CELL_MISSING_THRESHOLD
    expr, labels = expr[keep_cells.values], labels[keep_cells.values]
    print(f"After per-cell QC (<= {C.CELL_MISSING_THRESHOLD*100:.0f}% undetected): "
          f"{expr.shape[0]} cells")
    expr_frac = (expr > 0).sum(axis=0) / expr.shape[0]
    keep_genes = expr_frac >= C.GENE_EXPRESSED_MIN
    expr = expr.loc[:, keep_genes]
    print(f"After per-gene QC (>= {C.GENE_EXPRESSED_MIN*100:.0f}% detected): "
          f"{expr.shape[1]} genes")
    print("Class distribution (post-QC):")
    print(labels.value_counts().to_string(), "\n")
    return expr, labels


def _save_task(name, X_cl_tr, X_cl_te, X_q_tr, X_q_te, y_tr, y_te, feature_names):
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    fname = os.path.join(C.RESULTS_DIR, f"task_{name}.npz")
    np.savez(fname,
             X_train_classical=X_cl_tr, X_test_classical=X_cl_te,
             X_train_quantum=X_q_tr, X_test_quantum=X_q_te,
             y_train=y_tr, y_test=y_te, feature_names=np.array(feature_names))
    print(f"  saved {os.path.relpath(fname, C.ROOT)}  "
          f"(train={len(y_tr)}, test={len(y_te)}, genes={len(feature_names)})")


def build_and_save_tasks(expr, labels):
    label_to_int = {"nBM": 0, "Dx": 1, "TKI": 2}
    y_all = labels.map(label_to_int).values
    X_all = expr.values
    feature_names = expr.columns.tolist()
    tasks = {
        "diagnosis_binary": ([0, 1], {0: 0, 1: 1}),
        "therapy_binary":   ([1, 2], {1: 0, 2: 1}),
        "three_class":      ([0, 1, 2], {0: 0, 1: 1, 2: 2}),
    }
    for task_name, (class_ints, remap) in tasks.items():
        mask = np.isin(y_all, class_ints)
        X_task = X_all[mask]
        y_task = np.array([remap[v] for v in y_all[mask]])
        print(f"-- Task: {task_name} --")
        uniq, cnt = np.unique(y_task, return_counts=True)
        print(f"   samples: {X_task.shape[0]}, features: {X_task.shape[1]}, "
              f"class counts: {dict(zip(uniq.tolist(), cnt.tolist()))}")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_task, y_task, test_size=C.TEST_SIZE,
            random_state=C.RANDOM_STATE, stratify=y_task)
        scaler = StandardScaler()
        X_tr_cl = scaler.fit_transform(X_tr)
        X_te_cl = scaler.transform(X_te)
        n_comp = min(C.N_PCA, X_tr_cl.shape[1], X_tr_cl.shape[0])
        pca = PCA(n_components=n_comp, random_state=C.RANDOM_STATE)
        X_tr_pca = pca.fit_transform(X_tr_cl)
        X_te_pca = pca.transform(X_te_cl)
        lo, hi = X_tr_pca.min(axis=0), X_tr_pca.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        X_tr_q = np.clip(np.pi * (X_tr_pca - lo) / denom, 0, np.pi)
        X_te_q = np.clip(np.pi * (X_te_pca - lo) / denom, 0, np.pi)
        _save_task(task_name, X_tr_cl, X_te_cl, X_tr_q, X_te_q, y_tr, y_te, feature_names)
        print()


def main():
    download_data()
    expr_ct = load_expression_table(C.LOCAL_DATA)
    title_to_label = load_labels_from_matrix(C.LOCAL_MATRIX)
    expr_ct, labels = attach_labels(expr_ct, title_to_label)
    expr = convert_ct_to_expression(expr_ct)
    expr, labels = qc_filter(expr, labels)
    build_and_save_tasks(expr, labels)
    print("Preprocessing complete. Task files in results/.")
