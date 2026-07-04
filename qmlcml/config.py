"""Central configuration: paths, dataset constants, CV settings."""

import os

# --- paths -------------------------------------------------------------------
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(ROOT, "results")
METRICS_DIR  = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR  = os.path.join(ROOT, "figures")

# --- dataset -----------------------------------------------------------------
DATA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84507/suppl/"
    "GSE84507_non-normalized_data.txt.gz"
)
MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84507/matrix/"
    "GSE84507_series_matrix.txt.gz"
)
LOCAL_DATA   = os.path.join(ROOT, "GSE84507_non-normalized_data.txt.gz")
LOCAL_MATRIX = os.path.join(ROOT, "GSE84507_series_matrix.txt.gz")

# --- preprocessing constants (see preprocess.py for rationale) ---------------
LOD          = 40.0    # detection ceiling / failed-reaction sentinel (auto-verified)
N_PCA        = 8       # qubit count / PCA components for quantum encoding
TEST_SIZE    = 0.2
CELL_MISSING_THRESHOLD = 0.70
GENE_EXPRESSED_MIN     = 0.10

# --- evaluation --------------------------------------------------------------
N_SPLITS     = 5
RANDOM_STATE = 42

TASKS = ["diagnosis_binary", "therapy_binary", "three_class"]
TASK_TITLES = {
    "diagnosis_binary": "Diagnosis (nBM vs Dx)",
    "therapy_binary":   "Therapy response (Dx vs TKI)",
    "three_class":      "Three-class (nBM/Dx/TKI)",
}
CLASS_NAMES = {
    "diagnosis_binary": ["nBM", "Dx"],
    "therapy_binary":   ["Dx", "TKI"],
    "three_class":      ["nBM", "Dx", "TKI"],
}
