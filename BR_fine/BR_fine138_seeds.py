"""
BR + Random Forest -- 30 independent random-seed evaluation (no k-fold).

Companion script to HMCN_30seeds.py. Each of the two model scripts is run
independently, with its own draw of 30 seeds (per-model, NOT shared across
models -- this isolates each model's own resampling
variability rather than pairing splits for a paired head-to-head test).

Differences from baseline_fine138_kfold.py:
    1. No k-fold, no grid search. The single winning BR+RF config already
       established via the 5-fold sweep (RF_n_estimators=300_max_features=log2,
       confirmed in best_config_RF_fine138.csv and
       baseline_fine138_kfold_winning_configs.csv) is hardcoded and run once
       per seed, unchanged.
    2. Each seed draws its OWN single train/val/test split (80/10/10 via a
       two-stage IterativeStratification(order=2) on the 138 fine labels),
       not a partition of the dataset into folds -- splits across seeds are
       independent draws and may overlap each other.
    3. The same integer seed drives BOTH the split (random_state passed to
       the stratifier, plus the global numpy RNG skmultilearn falls back on
       for tie-breaking) AND the model's own internal randomness
       (RandomForestClassifier(random_state=seed)). Per-seed variance therefore mixes split variance and model-
       init/bootstrap variance rather than isolating either one.
    4. Restart-skip is keyed on `seed` alone (not fold/model/config), since
       there is only one model/config in this script.
    5. Thresholds are still calibrated on VAL only, evaluated on TEST --
       unchanged from the k-fold version.

Usage:
    Smoke test:  DEBUG=True in CONFIG (2 seeds, few labels).
    Full run (server, background):
        nohup python3 -u RF_BR_30seeds.py > /dev/null 2>&1 &
        (log file is the source of truth either way; /dev/null here is fine)
"""

import logging
import sys
import time
import random
import pickle
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score,
    confusion_matrix, accuracy_score, hamming_loss, jaccard_score
)
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.problem_transform import BinaryRelevance
import hmcn_eval  # shared eval utility -- keep hmcn_eval.py in the same directory

# ======================================================================
# CONFIG
# ======================================================================
DEBUG = False  # True -> 2 seeds, restricted labels, for a smoke test

DATA_PATH       = "hmcn_dataset.csv"
OUTPUT_DIR      = Path("outputs_RF_BR_30seeds")
LOG_DIR         = Path("logs")
SPLIT_CACHE_DIR = Path("seed_splits_RF_BR")   # independent from HMCN_30seeds.py's cache


# Sharding: the reported runs were produced by launching this script once per
# GPU with a distinct MASTER_SEED, then pooling the per-seed rows. MASTER_SEED
# and N_SEEDS are read from the environment so the shards are one script
# invoked several ways rather than several edited copies.
#     CUDA_VISIBLE_DEVICES=0 MASTER_SEED=42 N_SEEDS=30 nohup python3 -u RF_BR_30seeds.py &
# Verify the pooled result has as many DISTINCT seeds as claimed before
# reporting: shards sharing a MASTER_SEED draw identical seed lists, and
# pooling duplicates would understate the standard deviation.
MASTER_SEED = int(os.environ.get('MASTER_SEED', 42))
N_SEEDS     = int(os.environ.get('N_SEEDS', 30))
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# Winning config from the 5-fold sweep -- fixed here, not re-searched per seed.
RF_CONFIG = dict(
    n_estimators=300,
    max_features="log2",
    class_weight="balanced",
    n_jobs=-1,
)

if DEBUG:
    OUTPUT_DIR = Path("outputs_RF_BR_30seeds_debug")
    N_SEEDS = 2
    DEBUG_N_LABELS = 6

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)
SPLIT_CACHE_DIR.mkdir(exist_ok=True, parents=True)

# ======================================================================
# LOGGING (file + stdout, auto-timestamped, flushes every line)
# ======================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"RF_BR_30seeds_{'debug' if DEBUG else 'full'}_{timestamp}.log"

logger = logging.getLogger("RF_BR_30seeds")
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


def set_seed(seed):
    """Seed every RNG a given run depends on -- including the global numpy
    RNG that skmultilearn's IterativeStratification falls back on for
    tie-breaking regardless of the random_state passed to it."""
    random.seed(seed)
    np.random.seed(seed)



# ======================================================================
# THRESHOLD CALIBRATION (Section 5.1.3)
# ======================================================================
# Section 5.1.3 specifies a per-label sweep over [0.01, 0.99] in steps of
# 0.01, maximising F1 on the validation subset. hmcn_eval.find_optimal_
# thresholds uses 19 candidates in [0.05, 0.95] and is shared with other
# scripts, so the paper's grid is implemented locally here rather than by
# mutating the shared utility.
THRESHOLD_CANDIDATES = np.round(np.arange(0.01, 1.00, 0.01), 2)


def find_optimal_thresholds(probs, targets):
    """Per-label threshold maximising F1 on the validation subset."""
    n_labels = targets.shape[1]
    thresholds = np.full(n_labels, 0.5)
    for i in range(n_labels):
        if targets[:, i].sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in THRESHOLD_CANDIDATES:
            f1 = f1_score(targets[:, i], (probs[:, i] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def population_std(s):
    """Section 5.1.1 reports the POPULATION standard deviation (ddof=0).
    pandas' .std() defaults to ddof=1, the sample std -- overridden here."""
    return s.std(ddof=0)


# ======================================================================
# SEED GENERATION
# ======================================================================
def generate_seeds(master_seed=MASTER_SEED, n=N_SEEDS):
    rng = np.random.default_rng(master_seed)
    return rng.integers(0, 2**31 - 1, size=n).tolist()


# ======================================================================
# LOAD (full dataset, no split yet)
# ======================================================================
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {DATA_PATH}: {df.shape}")

    label_cols = [c for c in df.columns if c.startswith("fine_")]
    meta_cols  = [c for c in df.columns if c.startswith("meta_")]
    fp_cols    = [c for c in df.columns if c.startswith("MACCS_") or c.startswith("morgan_")]
    mordred_cols = [c for c in df.columns if c not in label_cols + meta_cols + ["SMILES"] + fp_cols]

    logger.info(f"Labels (fine_*)   : {len(label_cols)}")
    logger.info(f"Meta cols excluded: {len(meta_cols)}")
    logger.info(f"FP cols           : {len(fp_cols)}")
    logger.info(f"Mordred cols      : {len(mordred_cols)}")

    n_nan = df[fp_cols + mordred_cols].isna().sum().sum()
    logger.info(f"NaNs in features  : {n_nan}")
    df_clean = df.dropna(subset=fp_cols + mordred_cols).reset_index(drop=True)

    if len(df_clean) != len(df):
        raise RuntimeError(
            f"dropna removed {len(df) - len(df_clean)} rows -- this invalidates any "
            f"cached seed splits, which assume df's original row order is preserved. "
            f"Do not proceed without clearing seed_splits_RF_BR/ and re-deriving."
        )

    pos_counts = df_clean[label_cols].sum().sort_values()
    n_rare = (pos_counts < 10).sum()
    if n_rare:
        logger.warning(f"{n_rare} labels have <10 positive examples in the full set "
                        f"(rarest: {pos_counts.index[0]}={int(pos_counts.iloc[0])}). "
                        f"Expect noisy per-label metrics for these at small test sizes.")

    if DEBUG:
        label_cols = label_cols[:DEBUG_N_LABELS]
        logger.info(f"[DEBUG] restricted to {len(label_cols)} labels")

    return df_clean, label_cols, fp_cols, mordred_cols


# ======================================================================
# PER-SEED SPLIT (single 80/10/10 draw, not a k-fold partition)
# ======================================================================
def build_or_load_split(seed, n_rows, Y, cache_dir=SPLIT_CACHE_DIR,
                         train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    cache_path = cache_dir / f"seed_{seed}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    holdout_frac = val_ratio + test_ratio  # 0.20

    placeholder = np.arange(n_rows).reshape(-1, 1)
    np.random.seed(seed)
    stage1 = IterativeStratification(
        n_splits=2, order=2,
        sample_distribution_per_fold=[holdout_frac, 1 - holdout_frac],
    )
    train_idx, holdout_idx = next(stage1.split(placeholder, Y))

    np.random.seed(seed + 1)
    X_ho = np.arange(len(holdout_idx)).reshape(-1, 1)
    y_ho = Y[holdout_idx]
    rel_test_frac = test_ratio / holdout_frac  # 0.5 when val_ratio == test_ratio
    stage2 = IterativeStratification(
        n_splits=2, order=2,
        sample_distribution_per_fold=[rel_test_frac, 1 - rel_test_frac],
    )
    val_rel, test_rel = next(stage2.split(X_ho, y_ho))
    val_idx = holdout_idx[val_rel]
    test_idx = holdout_idx[test_rel]

    split = (train_idx, val_idx, test_idx)
    with open(cache_path, "wb") as f:
        pickle.dump(split, f)
    return split


# ======================================================================
# PER-SPLIT FEATURE PIPELINE (fit on that split's train portion only)
# ======================================================================
def build_split_features(X_fp_raw, X_mordred_raw, y, train_idx, val_idx, test_idx):
    Xfp_tr, Xfp_va, Xfp_te = X_fp_raw[train_idx], X_fp_raw[val_idx], X_fp_raw[test_idx]
    Xmo_tr, Xmo_va, Xmo_te = X_mordred_raw[train_idx], X_mordred_raw[val_idx], X_mordred_raw[test_idx]
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]

    vt_fp = VarianceThreshold(threshold=0)
    Xfp_tr = vt_fp.fit_transform(Xfp_tr)
    Xfp_va = vt_fp.transform(Xfp_va)
    Xfp_te = vt_fp.transform(Xfp_te)

    vt_mo = VarianceThreshold(threshold=0)
    Xmo_tr = vt_mo.fit_transform(Xmo_tr)
    Xmo_va = vt_mo.transform(Xmo_va)
    Xmo_te = vt_mo.transform(Xmo_te)

    scaler = StandardScaler()
    Xmo_tr = scaler.fit_transform(Xmo_tr)
    Xmo_va = scaler.transform(Xmo_va)
    Xmo_te = scaler.transform(Xmo_te)

    corr = np.abs(np.corrcoef(Xmo_tr.T))
    upper = np.triu(corr, k=1)
    drop = set()
    rows, cols = np.where(upper > 0.95)
    for r, c in zip(rows, cols):
        if c not in drop:
            drop.add(c)
    keep = [i for i in range(Xmo_tr.shape[1]) if i not in drop]
    Xmo_tr, Xmo_va, Xmo_te = Xmo_tr[:, keep], Xmo_va[:, keep], Xmo_te[:, keep]

    X_train = np.hstack([Xfp_tr, Xmo_tr])
    X_val   = np.hstack([Xfp_va, Xmo_va])
    X_test  = np.hstack([Xfp_te, Xmo_te])

    logger.info(f"  Split features: fp={Xfp_tr.shape[1]}, mordred={Xmo_tr.shape[1]} "
                f"(dropped {len(drop)} correlated) | "
                f"train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")

    return X_train, y_tr, X_val, y_va, X_test, y_te


# ======================================================================
# THRESHOLDS (on VAL, not train) + EVALUATION (on TEST)
# ======================================================================
def compute_fine_only_metrics(fine_probs, fine_true, fine_thresholds):
    """Same formulas as the fine-138 block of hmcn_eval.compute_all_metrics,
    copied rather than called directly because that function requires meta
    predictions the BR baseline doesn't produce. hier_violation_rate_138 is
    deliberately omitted (needs a parent/meta prediction to check against) --
    hmcn_eval.save_experiment fills it, and every meta_*/12 column, with NaN.
    """
    fine_pred = (fine_probs >= fine_thresholds[np.newaxis, :]).astype(int)
    valid_fine = [i for i in range(fine_true.shape[1]) if fine_true[:, i].sum() > 0]

    roc_auc_138 = float(np.mean([
        roc_auc_score(fine_true[:, i], fine_probs[:, i]) for i in valid_fine
    ]))
    pr_auc_138 = float(np.mean([
        average_precision_score(fine_true[:, i], fine_probs[:, i]) for i in valid_fine
    ]))

    return {
        "roc_auc_138"         : round(roc_auc_138, 4),
        "pr_auc_138"          : round(pr_auc_138, 4),
        "f1_macro_138"        : round(float(f1_score(fine_true, fine_pred, average="macro", zero_division=0)), 4),
        "f1_micro_138"        : round(float(f1_score(fine_true, fine_pred, average="micro", zero_division=0)), 4),
        "precision_macro_138" : round(float(precision_score(fine_true, fine_pred, average="macro", zero_division=0)), 4),
        "recall_macro_138"    : round(float(recall_score(fine_true, fine_pred, average="macro", zero_division=0)), 4),
        "precision_micro_138" : round(float(precision_score(fine_true, fine_pred, average="micro", zero_division=0)), 4),
        "recall_micro_138"    : round(float(recall_score(fine_true, fine_pred, average="micro", zero_division=0)), 4),
        "matched_accuracy_138": round(float(accuracy_score(fine_true, fine_pred)), 4),
        "hamming_loss_138"    : round(float(hamming_loss(fine_true, fine_pred)), 4),
        "jaccard_138"         : round(float(jaccard_score(fine_true, fine_pred, average="macro", zero_division=0)), 4),
        # hier_violation_rate_138 intentionally omitted -- see docstring
    }


def evaluate(seed, clf, X_val, y_val, X_test, y_test, label_cols):
    """clf is already fit. Thresholds are calibrated on VAL, metrics reported on TEST."""
    y_prob_val = np.array(clf.predict_proba(X_val).todense())
    y_prob_test = np.array(clf.predict_proba(X_test).todense())

    thr_array = find_optimal_thresholds(y_prob_val, y_val)
    thresholds = dict(zip(label_cols, thr_array))

    rows = []
    for i, label in enumerate(label_cols):
        yt, ypr = y_test[:, i], y_prob_test[:, i]
        yp = (ypr >= thresholds[label]).astype(int)

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        try:    roc = roc_auc_score(yt, ypr)
        except: roc = float("nan")
        try:    pr = average_precision_score(yt, ypr)
        except: pr = float("nan")
        try:    mcc = matthews_corrcoef(yt, yp)
        except: mcc = float("nan")

        rows.append({
            "Seed": seed, "Model": "RF_BR", "Label": label,
            "Threshold": round(float(thresholds[label]), 2),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Bal.Acc": round(balanced_accuracy_score(yt, yp), 3),
            "MCC": round(mcc, 3) if mcc == mcc else float("nan"),
            "F1": round(f1_score(yt, yp, zero_division=0), 3),
            "ROC_AUC": round(roc, 3),
            "PR_AUC": round(pr, 3),
            "Precision": round(precision_score(yt, yp, zero_division=0), 3),
            "Sensitivity": round(recall_score(yt, yp, zero_division=0), 3),
            "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else float("nan"), 3),
        })

    result_df = pd.DataFrame(rows)
    macro = result_df.drop(columns=["Seed", "Model", "Label"]).mean(numeric_only=True)
    logger.info(f"  [RF_BR] seed {seed} MACRO  F1={macro['F1']:.3f}  PR_AUC={macro['PR_AUC']:.3f}  "
                f"ROC_AUC={macro['ROC_AUC']:.3f}  Bal.Acc={macro['Bal.Acc']:.3f}  "
                f"Precision={macro['Precision']:.3f}")

    ablation_metrics = compute_fine_only_metrics(y_prob_test, y_test, thr_array)
    return result_df, ablation_metrics


def get_model(seed):
    return BinaryRelevance(
        classifier=RandomForestClassifier(random_state=seed, **RF_CONFIG),
        require_dense=[True, True],
    )


def run_seed(seed, X_train, y_train, X_val, y_val, X_test, y_test, label_cols, ablation_csv_path):
    logger.info(f"  [RF_BR] seed {seed}: fitting")
    clf = get_model(seed)

    t0 = time.time()
    clf.fit(X_train, y_train)
    logger.info(f"  [RF_BR] seed {seed}: fit done in {time.time() - t0:.1f}s")

    per_label_df, ablation_metrics = evaluate(seed, clf, X_val, y_val, X_test, y_test, label_cols)

    config = dict(
        experiment="RF_BR_30seeds", param_name="model",
        param_value="RF_n_estimators=300_max_features=log2",
        seed=seed,
        # global_dim/local_dim/dropout/lr/weight_decay/lambda_viol/beta/batch_size,
        # best_epoch/train_loss_at_best/val_meta_roc_auc, test_loss: not applicable
        # to sklearn models -- hmcn_eval.save_experiment NaN-fills these automatically.
    )
    hmcn_eval.save_experiment(config, ablation_metrics, csv_path=ablation_csv_path)

    return per_label_df


# ======================================================================
# MAIN
# ======================================================================
def main():
    df_clean, label_cols, fp_cols, mordred_cols = load_and_prepare()
    X_fp_raw = df_clean[fp_cols].values.astype(float)
    X_mordred_raw = df_clean[mordred_cols].values.astype(float)

    # Split stratification always uses the FULL 138-label set, even in DEBUG
    # mode with a restricted label_cols -- keeps the cached splits identical
    # to a full run's, rather than a debug-only variant.
    y_for_split = df_clean[[c for c in df_clean.columns if c.startswith("fine_")]].values
    y_full = df_clean[label_cols].values

    seeds = generate_seeds()
    if DEBUG:
        seeds = seeds[:N_SEEDS]
    logger.info(f"Seeds ({len(seeds)}): {seeds}")

    results_path = OUTPUT_DIR / "RF_BR_30seeds_per_label_results.csv"
    ablation_csv_path = OUTPUT_DIR / "RF_BR_30seeds_ablation_results.csv"

    # --- Restart-skip: which seeds are already done? ---
    completed = set()
    all_results = []
    if results_path.exists():
        prev = pd.read_csv(results_path)
        completed = set(prev["Seed"].unique().tolist())
        all_results.append(prev)
        logger.info(f"Found {len(completed)} completed seeds in '{results_path}' -- these will be skipped.")

    for i, seed in enumerate(seeds, 1):
        if seed in completed:
            logger.info(f"[{i}/{len(seeds)}] seed {seed}: already completed, skipping")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"[{i}/{len(seeds)}] SEED {seed}")
        logger.info(f"{'='*60}")

        set_seed(seed)
        train_idx, val_idx, test_idx = build_or_load_split(seed, len(df_clean), y_for_split)
        logger.info(f"  Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        X_train, y_train, X_val, y_val, X_test, y_test = build_split_features(
            X_fp_raw, X_mordred_raw, y_full, train_idx, val_idx, test_idx
        )

        try:
            res = run_seed(seed, X_train, y_train, X_val, y_val, X_test, y_test,
                            label_cols, ablation_csv_path)
            all_results.append(res)
            pd.concat(all_results, ignore_index=True).to_csv(results_path, index=False)
        except Exception:
            logger.exception(f"  seed {seed}: failed, continuing")

    if not all_results:
        logger.error("No run completed successfully.")
        return

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(results_path, index=False)

    metric_cols = ["Bal.Acc", "MCC", "F1", "ROC_AUC", "PR_AUC", "Precision", "Sensitivity", "Specificity"]

    # Per-seed macro average (mean across the 138 labels), one row per seed
    per_seed_macro = all_results_df.groupby(["Seed"])[metric_cols].mean().round(3)
    per_seed_macro.to_csv(OUTPUT_DIR / "RF_BR_30seeds_per_seed_macro.csv")

    # Mean +/- std across the 30 seeds -- single row, since there's only one config.
    summary = per_seed_macro[metric_cols].agg(["mean", population_std]).round(3)
    summary.index = ["mean", "std"]  # population std (ddof=0), Section 5.1.1
    summary.to_csv(OUTPUT_DIR / "RF_BR_30seeds_summary_mean_std.csv")
    logger.info("Mean +/- std across seeds:\n" + summary.to_string())

    if ablation_csv_path.exists():
        ablation_df = pd.read_csv(ablation_csv_path)
        exclude_cols = ["experiment", "param_name", "param_value", "seed",
                         "global_dim", "local_dim", "dropout", "lr", "weight_decay",
                         "lambda_viol", "beta", "batch_size",
                         "best_epoch", "train_loss_at_best", "val_meta_roc_auc"]
        metric_cols_ablation = [c for c in hmcn_eval._CSV_COLUMNS if c not in exclude_cols]
        ablation_summary = ablation_df[metric_cols_ablation].agg(["mean", population_std])
        ablation_summary.index = ["mean", "std"]  # population std (ddof=0)
        ablation_summary_path = OUTPUT_DIR / "RF_BR_30seeds_ablation_summary_mean_std.csv"
        ablation_summary.to_csv(ablation_summary_path)
        logger.info(f"Ablation-format mean +/- std summary "
                    f"(compare directly against HMCN_30seeds_ablation_summary_mean_std.csv): "
                    f"{ablation_summary_path}")

    logger.info(f"Done. Per-label results in {results_path}")
    logger.info(f"Done. Ablation-format results in {ablation_csv_path}")

    script_end_time = datetime.now()
    elapsed = script_end_time - SCRIPT_START_TIME
    logger.info(f"Script finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
