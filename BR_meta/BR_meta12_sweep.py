"""
Binary Relevance hyperparameter sweep on the 12 OR-aggregated metacategories.
Input: hmcn_dataset.csv
Output: outputs_meta12_sweep/
"""

import logging
import sys
import time
import random
import itertools
import pickle
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score,
    confusion_matrix, accuracy_score, hamming_loss, jaccard_score
)
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.problem_transform import BinaryRelevance
import hmcn_eval

if 'fold' not in hmcn_eval._CSV_COLUMNS:
    hmcn_eval._CSV_COLUMNS.insert(3, 'fold')
if 'model' not in hmcn_eval._CSV_COLUMNS:
    hmcn_eval._CSV_COLUMNS.insert(4, 'model')

# ======================================================================
# CONFIG
# ======================================================================
DEBUG = True

DATA_PATH    = "hmcn_dataset.csv"
OUTPUT_DIR   = Path("outputs_meta12_sweep")
LOG_DIR      = Path("logs")
SPLIT_DIR    = Path("sweep_splits")
RANDOM_STATE = 42

TEST_RATIO = 0.10
VAL_RATIO  = 0.10
K          = 5

REFERENCE_SPLIT_CACHE = SPLIT_DIR / "reference_split_seed42.pkl"
CV_FOLDS_CACHE        = SPLIT_DIR / "sweep_cv_folds_seed42.pkl"

DIAGNOSTIC_THRESHOLD = 0.5

MODELS_TO_RUN = ["LR", "RF", "XGB", "SVM", "KNN"]

MODEL_GRIDS = {
    "LR":  {"C": [0.001, 0.1, 1, 10, 100]},
    "RF":  {"n_estimators": [100, 300], "max_features": ["sqrt", "log2"]},
    "XGB": {"n_estimators": [100, 300], "max_depth": [3, 6]},
    "SVM": {"C": [0.1, 1, 10]},
    "KNN": {"n_neighbors": [5, 11, 21], "metric": ["euclidean", "cosine"]},
}

if DEBUG:
    OUTPUT_DIR = Path("outputs_meta12_sweep_debug")
    MODELS_TO_RUN = ["LR"]
    MODEL_GRIDS["LR"] = {"C": [0.1]}
    DEBUG_N_FOLDS = 1
    DEBUG_N_LABELS = 6

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)
SPLIT_DIR.mkdir(exist_ok=True, parents=True)

# ======================================================================
# REPRODUCIBILITY
# ======================================================================
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

set_seed(RANDOM_STATE)

# ======================================================================
# LOGGING
# ======================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"baseline_meta12_sweep_{'debug' if DEBUG else 'full'}_{timestamp}.log"

logger = logging.getLogger("baseline_meta12_sweep")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

fh = logging.FileHandler(log_path)
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

logger.info(f"DEBUG={DEBUG} | log file: {log_path}")
SCRIPT_START_TIME = datetime.now()
logger.info(f"Script started at: {SCRIPT_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")

# ======================================================================
# LOAD
# ======================================================================
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {DATA_PATH}: {df.shape}")

    label_cols = [c for c in df.columns if c.startswith("meta_")]
    fine_cols  = [c for c in df.columns if c.startswith("fine_")]
    fp_cols    = [c for c in df.columns if c.startswith("MACCS_") or c.startswith("morgan_")]
    mordred_cols = [c for c in df.columns if c not in label_cols + fine_cols + ["SMILES"] + fp_cols]

    logger.info(f"Labels (meta_*)   : {len(label_cols)}")
    logger.info(f"Fine cols excluded: {len(fine_cols)}")
    logger.info(f"FP cols           : {len(fp_cols)}")
    logger.info(f"Mordred cols      : {len(mordred_cols)}")

    n_nan = df[fp_cols + mordred_cols].isna().sum().sum()
    logger.info(f"NaNs in features  : {n_nan}")
    df_clean = df.dropna(subset=fp_cols + mordred_cols).reset_index(drop=True)

    if len(df_clean) != len(df):
        raise RuntimeError(
            f"dropna removed {len(df) - len(df_clean)} rows -- this invalidates the "
            f"cached split indices. Do not proceed without re-deriving the splits."
        )

    if DEBUG:
        label_cols = label_cols[:DEBUG_N_LABELS]
        logger.info(f"[DEBUG] restricted to {len(label_cols)} labels")

    return df_clean, label_cols, fp_cols, mordred_cols

# ======================================================================
# SPLITS
# ======================================================================
def build_or_load_reference_split(Y, seed=RANDOM_STATE, cache_path=REFERENCE_SPLIT_CACHE):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            split = pickle.load(f)
        logger.info(f"Loaded cached reference split from '{cache_path}'.")
        return split

    n = len(Y)
    placeholder = np.arange(n).reshape(-1, 1)

    np.random.seed(seed)
    s1 = IterativeStratification(
        n_splits=2, order=2,
        sample_distribution_per_fold=[TEST_RATIO, 1.0 - TEST_RATIO],
    )
    trainval_idx, test_idx = next(s1.split(placeholder, Y))

    np.random.seed(seed + 1)
    val_frac_of_remainder = VAL_RATIO / (1.0 - TEST_RATIO)
    s2 = IterativeStratification(
        n_splits=2, order=2,
        sample_distribution_per_fold=[val_frac_of_remainder, 1.0 - val_frac_of_remainder],
    )
    tv_placeholder = np.arange(len(trainval_idx)).reshape(-1, 1)
    train_rel, val_rel = next(s2.split(tv_placeholder, Y[trainval_idx]))

    split = (trainval_idx[train_rel], trainval_idx[val_rel], test_idx)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(split, f)
    logger.info(f"Computed reference split and cached to '{cache_path}'.")
    return split

def build_or_load_cv_folds(train_idx, Y, k=K, seed=RANDOM_STATE, cache_path=CV_FOLDS_CACHE):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            folds = pickle.load(f)
        logger.info(f"Loaded cached {len(folds)}-fold CV split from '{cache_path}'.")
        return folds

    np.random.seed(seed + 100)
    placeholder = np.arange(len(train_idx)).reshape(-1, 1)
    y_train = Y[train_idx]

    splitter = IterativeStratification(n_splits=k, order=2)
    folds = [
        (train_idx[fit_rel], train_idx[hold_rel])
        for fit_rel, hold_rel in splitter.split(placeholder, y_train)
    ]

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(folds, f)
    logger.info(f"Computed {k}-fold CV split and cached to '{cache_path}'.")
    return folds

def assert_splits_disjoint(train_idx, val_idx, test_idx, folds):
    train_set, val_set, test_set = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Reference split blocks overlap.")

    held_out_union = set()
    for fit_idx, hold_idx in folds:
        fit_set, hold_set = set(fit_idx.tolist()), set(hold_idx.tolist())
        if fit_set & hold_set:
            raise RuntimeError("A CV fold's fit and held-out portions overlap.")
        if not (fit_set | hold_set) <= train_set:
            raise RuntimeError("A CV fold reaches outside the training subset.")
        if held_out_union & hold_set:
            raise RuntimeError("A molecule is held out in more than one CV fold.")
        held_out_union |= hold_set

    if held_out_union != train_set:
        raise RuntimeError("The CV folds do not cover the training subset exactly once.")

    logger.info(f"Split check passed: train={len(train_set)} val={len(val_set)} test={len(test_set)}")

# ======================================================================
# PER-FOLD FEATURE PIPELINE 
# ======================================================================
def build_fold_features(X_fp_raw, X_mordred_raw, y, fit_idx, hold_idx):
    Xfp_fit, Xfp_hold = X_fp_raw[fit_idx], X_fp_raw[hold_idx]
    Xmo_fit, Xmo_hold = X_mordred_raw[fit_idx], X_mordred_raw[hold_idx]
    y_fit, y_hold = y[fit_idx], y[hold_idx]

    vt_fp = VarianceThreshold(threshold=0)
    Xfp_fit = vt_fp.fit_transform(Xfp_fit)
    Xfp_hold = vt_fp.transform(Xfp_hold)

    vt_mo = VarianceThreshold(threshold=0)
    Xmo_fit = vt_mo.fit_transform(Xmo_fit)
    Xmo_hold = vt_mo.transform(Xmo_hold)

    scaler = StandardScaler()
    Xmo_fit = scaler.fit_transform(Xmo_fit)
    Xmo_hold = scaler.transform(Xmo_hold)

    corr = np.abs(np.corrcoef(Xmo_fit.T))
    upper = np.triu(corr, k=1)
    drop = set()
    rows, cols = np.where(upper > 0.95)
    for r, c in zip(rows, cols):
        if c not in drop:
            drop.add(c)
    keep = [i for i in range(Xmo_fit.shape[1]) if i not in drop]
    Xmo_fit, Xmo_hold = Xmo_fit[:, keep], Xmo_hold[:, keep]

    X_fit = np.hstack([Xfp_fit, Xmo_fit])
    X_hold = np.hstack([Xfp_hold, Xmo_hold])

    logger.info(f"  Fold features: fp={Xfp_fit.shape[1]}, mordred={Xmo_fit.shape[1]} "
                f"(dropped {len(drop)} correlated) | "
                f"fit={X_fit.shape[0]} heldout={X_hold.shape[0]}")

    return X_fit, y_fit, X_hold, y_hold

# ======================================================================
# EVALUATION
# ======================================================================
def compute_meta_only_metrics(meta_probs, meta_true, Y_fit, threshold=DIAGNOSTIC_THRESHOLD):
    meta_pred = (meta_probs >= threshold).astype(int)
    valid_meta = [i for i in range(meta_true.shape[1]) if meta_true[:, i].sum() > 0]

    roc_auc_12 = float(np.mean([
        roc_auc_score(meta_true[:, i], meta_probs[:, i]) for i in valid_meta
    ]))
    pr_auc_12 = float(np.mean([
        average_precision_score(meta_true[:, i], meta_probs[:, i]) for i in valid_meta
    ]))
    recall_macro_12 = float(recall_score(meta_true, meta_pred, average='macro', zero_division=0))
    balanced_accuracy_12 = float(np.mean([
        balanced_accuracy_score(meta_true[:, i], meta_pred[:, i]) for i in valid_meta
    ]))

    return {
        'roc_auc_12'               : round(roc_auc_12, 4),
        'pr_auc_12'                : round(pr_auc_12, 4),
        'f1_macro_12'              : round(float(f1_score(meta_true, meta_pred, average='macro', zero_division=0)), 4),
        'instance_f1_12'           : round(float(f1_score(meta_true, meta_pred, average='samples', zero_division=0)), 4),
        'balanced_accuracy_12'     : round(balanced_accuracy_12, 4),
        'matched_accuracy_12'      : round(float(accuracy_score(meta_true, meta_pred)), 4),
        'sensitivity_macro_12'     : round(recall_macro_12, 4), 
        'specificity_macro_12'     : round(hmcn_eval._macro_specificity(meta_true, meta_pred), 4),
        'precision_macro_12'       : round(float(precision_score(meta_true, meta_pred, average='macro', zero_division=0)), 4),
        'recall_macro_12'          : round(recall_macro_12, 4),
        'hamming_loss_12'          : round(float(hamming_loss(meta_true, meta_pred)), 4),
        'jaccard_12'               : round(float(jaccard_score(meta_true, meta_pred, average='macro', zero_division=0)), 4),
        'hier_violation_rate_12'   : 0.0,
        'label_cooc_consistency_12': round(hmcn_eval._label_cooccurrence_consistency(meta_pred, meta_true, Y_fit), 4),
    }

def evaluate(name, param_value, fold_i, clf, y_fit, X_hold, y_hold, label_cols):
    y_prob = np.array(clf.predict_proba(X_hold).todense())

    rows = []
    for i, label in enumerate(label_cols):
        yt, ypr = y_hold[:, i], y_prob[:, i]
        yp = (ypr >= DIAGNOSTIC_THRESHOLD).astype(int)

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        try:    roc = roc_auc_score(yt, ypr)
        except: roc = float("nan")
        try:    pr = average_precision_score(yt, ypr)
        except: pr = float("nan")
        try:    mcc = matthews_corrcoef(yt, yp)
        except: mcc = float("nan")

        rows.append({
            "Fold": fold_i, "Model": name, "Config": param_value, "Label": label,
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Bal.Acc": round(balanced_accuracy_score(yt, yp), 3),
            "MCC": round(mcc, 3) if mcc == mcc else float("nan"),
            "F1_at_0.5": round(f1_score(yt, yp, zero_division=0), 3),
            "ROC_AUC": round(roc, 3),
            "PR_AUC": round(pr, 3),
            "Precision_at_0.5": round(precision_score(yt, yp, zero_division=0), 3),
            "Sensitivity_at_0.5": round(recall_score(yt, yp, zero_division=0), 3),
            "Specificity_at_0.5": round(tn / (tn + fp) if (tn + fp) > 0 else float("nan"), 3),
        })

    result_df = pd.DataFrame(rows)
    macro = result_df.drop(columns=["Fold", "Model", "Config", "Label"]).mean(numeric_only=True)
    logger.info(f"  [{name}] fold {fold_i} [{param_value}] HELD-OUT MACRO  "
                f"PR_AUC={macro['PR_AUC']:.3f} (selection metric)  "
                f"ROC_AUC={macro['ROC_AUC']:.3f}  "
                f"F1@0.5={macro['F1_at_0.5']:.3f}  Bal.Acc={macro['Bal.Acc']:.3f}")

    ablation_metrics = compute_meta_only_metrics(y_prob, y_hold, y_fit)
    return result_df, ablation_metrics

# ======================================================================
# MODEL BUILDERS
# ======================================================================
def _make_auto_weight_xgb(**kwargs):
    from xgboost import XGBClassifier

    class _AutoWeightXGB(XGBClassifier):
        def fit(self, X, y, **fit_kwargs):
            y_arr = np.asarray(y).ravel()
            n_pos = int(y_arr.sum())
            n_neg = len(y_arr) - n_pos
            self.scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            return super().fit(X, y_arr, **fit_kwargs)

    return _AutoWeightXGB(**kwargs)

def get_model(name, cfg):
    if name == "LR":
        from sklearn.linear_model import LogisticRegression
        return BinaryRelevance(
            classifier=LogisticRegression(class_weight="balanced", solver="saga", max_iter=300,
                                          random_state=RANDOM_STATE, **cfg), 
            require_dense=[True, True],
        )
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return BinaryRelevance(
            classifier=RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                              n_jobs=-1, **cfg),
            require_dense=[True, True],
        )
    if name == "XGB":
        return BinaryRelevance(
            classifier=_make_auto_weight_xgb(eval_metric="logloss", random_state=RANDOM_STATE,
                                             n_jobs=-1, **cfg),
            require_dense=[True, True],
        )
    if name == "SVM":
        from sklearn.svm import SVC
        return BinaryRelevance(
            classifier=SVC(class_weight="balanced", kernel="rbf", probability=True,
                           random_state=RANDOM_STATE, **cfg),
            require_dense=[True, True],
        )
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return BinaryRelevance(classifier=KNeighborsClassifier(n_jobs=-1, **cfg),
                               require_dense=[True, True])
    raise ValueError(name)

def run_model(name, cfg, param_value, fold_i, X_fit, y_fit, X_hold, y_hold,
              label_cols, ablation_csv_path):
    logger.info(f"  [{name}] fold {fold_i}: fitting {param_value}")
    clf = get_model(name, cfg)

    t0 = time.time()
    clf.fit(X_fit, y_fit)
    logger.info(f"  [{name}] fold {fold_i}: fit done in {time.time() - t0:.1f}s")

    per_label_df, ablation_metrics = evaluate(name, param_value, fold_i, clf,
                                              y_fit, X_hold, y_hold, label_cols)

    config = dict(
        experiment='baseline_meta12_sweep', param_name='model', param_value=param_value,
        fold=fold_i, model=name, seed=RANDOM_STATE,
    )
    hmcn_eval.save_experiment(config, ablation_metrics, csv_path=ablation_csv_path)

    return per_label_df

def expand_grid(grid_dict):
    keys = list(grid_dict.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid_dict.values())]

def format_param_value(name, cfg):
    return name + "_" + "_".join(f"{k}={v}" for k, v in cfg.items())

# ======================================================================
# MAIN
# ======================================================================
def main():
    df_clean, label_cols, fp_cols, mordred_cols = load_and_prepare()
    X_fp_raw = df_clean[fp_cols].values.astype(float)
    X_mordred_raw = df_clean[mordred_cols].values.astype(float)
    y_full = df_clean[label_cols].values

    y_for_split = df_clean[[c for c in df_clean.columns if c.startswith("fine_")]].values

    train_idx, val_idx, test_idx = build_or_load_reference_split(y_for_split)
    folds = build_or_load_cv_folds(train_idx, y_for_split)
    assert_splits_disjoint(train_idx, val_idx, test_idx, folds)

    if DEBUG:
        folds = folds[:DEBUG_N_FOLDS]
        logger.info(f"[DEBUG] restricted to {len(folds)} fold(s)")

    results_path = OUTPUT_DIR / "baseline_meta12_sweep_all_results.csv"
    ablation_csv_path = OUTPUT_DIR / "baseline_meta12_sweep_ablation_results.csv"

    runs = [(name, cfg, format_param_value(name, cfg))
            for name in MODELS_TO_RUN for cfg in expand_grid(MODEL_GRIDS[name])]
    logger.info(f"Configs total: {len(runs)} | Folds: {len(folds)} | "
                f"Total runs: {len(runs) * len(folds)}")

    completed = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        completed = set(zip(prev["Fold"], prev["Model"], prev["Config"]))
        logger.info(f"Found {len(completed)} completed (fold, model, config) triples in "
                    f"'{results_path}' -- these will be skipped.")

    all_results = []
    if results_path.exists():
        all_results.append(pd.read_csv(results_path))

    for fold_i, (fit_idx, hold_idx) in enumerate(folds):
        logger.info(f"{'='*60}")
        logger.info(f"FOLD {fold_i+1}/{len(folds)}  "
                    f"({len(fit_idx)} fit / {len(hold_idx)} held out)")
        logger.info(f"{'='*60}")

        runs_remaining = [r for r in runs if (fold_i, r[0], r[2]) not in completed]
        if not runs_remaining:
            logger.info(f"  All {len(runs)} runs already completed for fold {fold_i} -- skipping.")
            continue

        X_fit, y_fit, X_hold, y_hold = build_fold_features(
            X_fp_raw, X_mordred_raw, y_full, fit_idx, hold_idx
        )

        for i, (name, cfg, param_value) in enumerate(runs, 1):
            if (fold_i, name, param_value) in completed:
                logger.info(f"  [{i}/{len(runs)}] {param_value}: already completed, skipping")
                continue
            try:
                res = run_model(name, cfg, param_value, fold_i, X_fit, y_fit, X_hold, y_hold,
                                label_cols, ablation_csv_path)
                all_results.append(res)
                pd.concat(all_results, ignore_index=True).to_csv(results_path, index=False)
            except Exception:
                logger.exception(f"  [{i}/{len(runs)}] {param_value}: failed, continuing")

    if not all_results:
        logger.error("No run completed successfully.")
        return

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(results_path, index=False)

    metric_cols = ["Bal.Acc", "MCC", "F1_at_0.5", "ROC_AUC", "PR_AUC",
                   "Precision_at_0.5", "Sensitivity_at_0.5", "Specificity_at_0.5"]
    metric_cols = [c for c in metric_cols if c in all_results_df.columns]

    per_fold_macro = all_results_df.groupby(["Fold", "Model", "Config"])[metric_cols].mean().round(3)
    per_fold_macro.to_csv(OUTPUT_DIR / "baseline_meta12_sweep_per_fold_macro.csv")

    summary = all_results_df.groupby(["Fold", "Model", "Config"])[metric_cols].mean() \
                            .groupby(["Model", "Config"]).agg(["mean", "std"]).round(3)
    summary.to_csv(OUTPUT_DIR / "baseline_meta12_sweep_summary_mean_std.csv")
    logger.info("Mean +/- std across CV folds, per (Model, Config):\n" + summary.to_string())

    if ablation_csv_path.exists():
        ablation_df = pd.read_csv(ablation_csv_path)
        exclude_cols = ['experiment', 'param_name', 'param_value', 'fold', 'model',
                        'global_dim', 'local_dim', 'dropout', 'lr', 'weight_decay',
                        'lambda_viol', 'beta', 'batch_size', 'seed',
                        'best_epoch', 'train_loss_at_best', 'val_meta_roc_auc']
        metric_cols_ablation = [c for c in hmcn_eval._CSV_COLUMNS if c not in exclude_cols]
        ablation_summary = ablation_df.groupby(['model', 'param_value'])[metric_cols_ablation].agg(['mean', 'std'])
        ablation_summary_path = OUTPUT_DIR / "baseline_meta12_sweep_ablation_summary_mean_std.csv"
        ablation_summary.to_csv(ablation_summary_path)
        logger.info(f"Ablation-format mean +/- std summary: {ablation_summary_path}")

        mean_only = ablation_df.groupby(['model', 'param_value'])[metric_cols_ablation].mean()
        winners = mean_only.loc[mean_only.groupby('model')['pr_auc_12'].idxmax()]
        winners_path = OUTPUT_DIR / "baseline_meta12_sweep_winning_configs.csv"
        winners.to_csv(winners_path)
        logger.info("Winning config per base learner:\n" + winners[['pr_auc_12', 'roc_auc_12', 'f1_macro_12']].to_string())
        logger.info(f"Winning configs saved to: {winners_path}")

    logger.info(f"Done. Per-label results in {results_path}")
    logger.info(f"Done. Ablation-format results in {ablation_csv_path}")

    script_end_time = datetime.now()
    elapsed = script_end_time - SCRIPT_START_TIME
    logger.info(f"Script finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed time: {elapsed}")

if __name__ == "__main__":
    main()