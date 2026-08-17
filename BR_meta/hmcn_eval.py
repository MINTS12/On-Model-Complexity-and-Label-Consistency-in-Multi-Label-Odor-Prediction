"""
hmcn_eval.py
------------
Shared evaluation and result-logging utility for all HMCN-F ablation experiments.

Usage
-----
from hmcn_eval import compute_all_metrics, save_experiment, find_optimal_thresholds

# Calibrate thresholds on val set
fine_thresholds = find_optimal_thresholds(fp_val, ft_val)
meta_thresholds = find_optimal_thresholds(mp_val, mt_val)

# Compute test loss (one forward pass over test_loader)
test_loss = compute_test_loss(model, test_loader, violation_pairs, LAMBDA_VIOL, device)

# Compute all metrics
metrics = compute_all_metrics(
    fine_probs=fp_test, fine_true=ft_test,
    meta_probs=mp_test, meta_true=mt_test,
    meta_thresholds=meta_thresholds,
    fine_thresholds=fine_thresholds,
    violation_pairs=violation_pairs,
    meta_names=meta_names,
    fine_names=fine_names,
    Y2_train=Y2_train,
    test_loss=test_loss_value,
)

# Build config dict and save
config = dict(
    experiment='beta_ablation', param_name='beta', param_value=0.3,
    global_dim=128, local_dim=64, dropout=0.47,
    lr=1e-4, weight_decay=1e-4, lambda_viol=0.1, beta=0.3,
    batch_size=32, seed=42,
    best_epoch=56, train_loss_at_best=1.234, val_meta_roc_auc=0.808,
)
save_experiment(config, metrics, csv_path='hmcn_ablation_results.csv')
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score,
    hamming_loss,
    jaccard_score,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_thresholds(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Per-label threshold optimisation on a held-out set (val).
    Sweeps 19 candidates in [0.01, 0.99] and picks the one maximising F1.
    Labels with no positive examples keep threshold = 0.5.
    """
    candidates = np.round(np.arange(0.01, 1.00, 0.01), 2)
    thresholds  = np.full(targets.shape[1], 0.5)
    for i in range(targets.shape[1]):
        if targets[:, i].sum() == 0:
            continue
        best_f1, best_tau = 0.0, 0.5
        for tau in candidates:
            f1 = f1_score(
                targets[:, i],
                (probs[:, i] >= tau).astype(int),
                zero_division=0,
            )
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        thresholds[i] = best_tau
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Test loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_test_loss(model, test_loader, violation_pairs, lambda_viol, device) -> float:
    """
    One forward pass over the test set to compute hmcn_loss.
    Requires the loss helpers (binary_cross_entropy, hierarchical_violation_penalty,
    hmcn_loss) to be defined in the calling notebook — passed implicitly via closure.
    Instead, we redefine them here so this function is fully self-contained.
    """
    def _bce(P, Y, eps=1e-7):
        P = torch.clamp(P, eps, 1 - eps)
        return -torch.mean(Y * torch.log(P) + (1 - Y) * torch.log(1 - P))

    def _viol(P_L1, P_L2, pairs):
        total = torch.tensor(0.0, device=P_L1.device)
        for fi, mi in pairs:
            v = torch.clamp(P_L1[:, fi] - P_L2[:, mi], min=0.0)
            total = total + torch.mean(v ** 2)
        return total / max(len(pairs), 1)

    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for X_batch, Y1_batch, Y2_batch in test_loader:
            X_batch  = X_batch.to(device)
            Y1_batch = Y1_batch.to(device)
            Y2_batch = Y2_batch.to(device)
            P_F, P_L1, P_L2, P_G = model(X_batch)
            Y_global = torch.cat([Y1_batch, Y2_batch], dim=1)
            loss = (_bce(P_L1, Y1_batch) + _bce(P_L2, Y2_batch)
                    + _bce(P_G, Y_global)
                    + lambda_viol * _viol(P_L1, P_L2, violation_pairs))
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _macro_specificity(targets: np.ndarray, preds: np.ndarray) -> float:
    """Macro-averaged specificity (TNR) across labels."""
    specs = []
    for i in range(targets.shape[1]):
        tn, fp, fn, tp = confusion_matrix(
            targets[:, i], preds[:, i], labels=[0, 1]
        ).ravel()
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specs))


def _hierarchical_violation_rate(
    fine_binary: np.ndarray,
    meta_binary: np.ndarray,
    violation_pairs: list,
) -> float:
    """
    Fraction of child-positive predictions where the parent is negative.

        ViolationRate = Σ 1[ŷ_child=1 ∧ ŷ_parent=0]
                        ─────────────────────────────
                             Σ 1[ŷ_child=1]

    Lower is better. 0 = hierarchy perfectly respected.
    """
    total_child_pos  = 0
    total_violations = 0
    for fi, mi in violation_pairs:
        child_pos    = fine_binary[:, fi]
        parent_pos   = meta_binary[:, mi]
        total_child_pos  += int(child_pos.sum())
        total_violations += int(((child_pos == 1) & (parent_pos == 0)).sum())
    return total_violations / total_child_pos if total_child_pos > 0 else 0.0


def _cooccurrence_matrix(binary: np.ndarray) -> np.ndarray:
    """rho[i,j] = P(label_j=1 | label_i=1). Shape: (n_labels, n_labels)."""
    n   = binary.shape[1]
    rho = np.zeros((n, n))
    for i in range(n):
        mask = binary[:, i] == 1
        if mask.sum() == 0:
            continue
        rho[i] = binary[mask].mean(axis=0)
    return rho


def _label_cooccurrence_consistency(
    meta_binary_pred: np.ndarray,
    meta_binary_true: np.ndarray,
    Y2_train: np.ndarray,
    threshold: float = 0.10,
) -> float:
    """
    MAE between predicted and true co-occurrence rates, restricted to pairs
    that co-occur meaningfully in the training set (rho_train > threshold).
    Lower is better. Uses training co-occurrence as reference to avoid
    evaluating on pairs that are rare in the test set by chance.
    """
    rho_train = _cooccurrence_matrix(Y2_train.astype(int))
    rho_true  = _cooccurrence_matrix(meta_binary_true.astype(int))
    rho_pred  = _cooccurrence_matrix(meta_binary_pred.astype(int))
    n = rho_train.shape[0]
    errors = [
        abs(rho_pred[i, j] - rho_true[i, j])
        for i in range(n) for j in range(n)
        if i != j and rho_train[i, j] > threshold
    ]
    return float(np.mean(errors)) if errors else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    fine_probs:       np.ndarray,   # (N, 138) — P_F[:, :n_fine]
    fine_true:        np.ndarray,   # (N, 138)
    meta_probs:       np.ndarray,   # (N, 12)  — P_F[:, n_fine:]
    meta_true:        np.ndarray,   # (N, 12)
    meta_thresholds:  np.ndarray,   # (12,)  — calibrated on val
    fine_thresholds:  np.ndarray,   # (138,) — calibrated on val
    violation_pairs:  list,         # [(fine_idx, meta_idx), ...]
    meta_names:       list,         # 12 strings
    fine_names:       list,         # 138 strings
    Y2_train:         np.ndarray,   # (N_train, 12) — for cooc reference
    test_loss:        float,
) -> dict:
    """
    Computes all metrics for one experiment run and returns them as a flat dict.

    Fine-label metrics (138)          Metacategory metrics (12)
    ────────────────────────          ─────────────────────────
    roc_auc_138                       roc_auc_12
    pr_auc_138                        pr_auc_12
    f1_macro_138                      f1_macro_12
    f1_micro_138                      instance_f1_12
    precision_macro_138               balanced_accuracy_12
    recall_macro_138                  matched_accuracy_12
    precision_micro_138               sensitivity_macro_12
    recall_micro_138                  specificity_macro_12
    matched_accuracy_138              precision_macro_12
    hamming_loss_138                  recall_macro_12
    jaccard_138                       hier_violation_rate_12
    hier_violation_rate_138           label_cooc_consistency_12
                                      hamming_loss_12
                                      jaccard_12
    Other
    ─────
    test_loss
    """

    # Binary predictions using val-calibrated per-label thresholds
    fine_pred = (fine_probs >= fine_thresholds[np.newaxis, :]).astype(int)
    meta_pred = (meta_probs >= meta_thresholds[np.newaxis, :]).astype(int)

    # ── Fine (138) ────────────────────────────────────────────────────────────
    valid_fine = [i for i in range(fine_true.shape[1]) if fine_true[:, i].sum() > 0]

    roc_auc_138 = float(np.mean([
        roc_auc_score(fine_true[:, i], fine_probs[:, i]) for i in valid_fine
    ]))
    pr_auc_138 = float(np.mean([
        average_precision_score(fine_true[:, i], fine_probs[:, i]) for i in valid_fine
    ]))

    f1_macro_138        = float(f1_score(fine_true, fine_pred, average='macro',  zero_division=0))
    f1_micro_138        = float(f1_score(fine_true, fine_pred, average='micro',  zero_division=0))
    precision_macro_138 = float(precision_score(fine_true, fine_pred, average='macro', zero_division=0))
    recall_macro_138    = float(recall_score(fine_true, fine_pred, average='macro',    zero_division=0))
    precision_micro_138 = float(precision_score(fine_true, fine_pred, average='micro', zero_division=0))
    recall_micro_138    = float(recall_score(fine_true, fine_pred, average='micro',    zero_division=0))
    matched_accuracy_138 = float(accuracy_score(fine_true, fine_pred))
    hamming_loss_138    = float(hamming_loss(fine_true, fine_pred))
    jaccard_138         = float(jaccard_score(fine_true, fine_pred, average='macro', zero_division=0))
    hier_violation_rate_138 = _hierarchical_violation_rate(fine_pred, meta_pred, violation_pairs)

    # ── Meta (12) ─────────────────────────────────────────────────────────────
    valid_meta = [i for i in range(meta_true.shape[1]) if meta_true[:, i].sum() > 0]

    roc_auc_12 = float(np.mean([
        roc_auc_score(meta_true[:, i], meta_probs[:, i]) for i in valid_meta
    ]))
    pr_auc_12 = float(np.mean([
        average_precision_score(meta_true[:, i], meta_probs[:, i]) for i in valid_meta
    ]))

    f1_macro_12         = float(f1_score(meta_true, meta_pred, average='macro',   zero_division=0))
    instance_f1_12      = float(f1_score(meta_true, meta_pred, average='samples', zero_division=0))
    precision_macro_12  = float(precision_score(meta_true, meta_pred, average='macro', zero_division=0))
    recall_macro_12     = float(recall_score(meta_true, meta_pred, average='macro',    zero_division=0))
    matched_accuracy_12 = float(accuracy_score(meta_true, meta_pred))
    hamming_loss_12     = float(hamming_loss(meta_true, meta_pred))
    jaccard_12          = float(jaccard_score(meta_true, meta_pred, average='macro', zero_division=0))
    sensitivity_macro_12 = recall_macro_12   # explicit alias
    specificity_macro_12 = _macro_specificity(meta_true, meta_pred)

    balanced_accuracy_12 = float(np.mean([
        balanced_accuracy_score(meta_true[:, i], meta_pred[:, i])
        for i in valid_meta
    ]))

    hier_violation_rate_12    = 0.0   # no parent above meta — sanity check
    label_cooc_consistency_12 = _label_cooccurrence_consistency(
        meta_pred, meta_true, Y2_train
    )

    return {
        # fine (138)
        'roc_auc_138'             : round(roc_auc_138,              4),
        'pr_auc_138'              : round(pr_auc_138,                4),
        'f1_macro_138'            : round(f1_macro_138,              4),
        'f1_micro_138'            : round(f1_micro_138,              4),
        'precision_macro_138'     : round(precision_macro_138,       4),
        'recall_macro_138'        : round(recall_macro_138,          4),
        'precision_micro_138'     : round(precision_micro_138,       4),
        'recall_micro_138'        : round(recall_micro_138,          4),
        'matched_accuracy_138'    : round(matched_accuracy_138,      4),
        'hamming_loss_138'        : round(hamming_loss_138,          4),
        'jaccard_138'             : round(jaccard_138,               4),
        'hier_violation_rate_138' : round(hier_violation_rate_138,   4),
        # meta (12)
        'roc_auc_12'              : round(roc_auc_12,                4),
        'pr_auc_12'               : round(pr_auc_12,                 4),
        'f1_macro_12'             : round(f1_macro_12,               4),
        'instance_f1_12'          : round(instance_f1_12,            4),
        'balanced_accuracy_12'    : round(balanced_accuracy_12,      4),
        'matched_accuracy_12'     : round(matched_accuracy_12,       4),
        'sensitivity_macro_12'    : round(sensitivity_macro_12,      4),
        'specificity_macro_12'    : round(specificity_macro_12,      4),
        'precision_macro_12'      : round(precision_macro_12,        4),
        'recall_macro_12'         : round(recall_macro_12,           4),
        'hamming_loss_12'         : round(hamming_loss_12,           4),
        'jaccard_12'              : round(jaccard_12,                4),
        'hier_violation_rate_12'  : round(hier_violation_rate_12,    4),
        'label_cooc_consistency_12': round(label_cooc_consistency_12, 4),
        # loss
        'test_loss'               : round(float(test_loss),          4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    # identification
    'experiment', 'param_name', 'param_value',
    # full config
    'global_dim', 'local_dim', 'dropout',
    'lr', 'weight_decay', 'lambda_viol', 'beta',
    'batch_size', 'seed',
    # training outcome
    'best_epoch', 'train_loss_at_best', 'val_meta_roc_auc',
    # fine (138)
    'roc_auc_138', 'pr_auc_138',
    'f1_macro_138', 'f1_micro_138',
    'precision_macro_138', 'recall_macro_138',
    'precision_micro_138', 'recall_micro_138',
    'matched_accuracy_138', 'hamming_loss_138', 'jaccard_138',
    'hier_violation_rate_138',
    # meta (12)
    'roc_auc_12', 'pr_auc_12',
    'f1_macro_12', 'instance_f1_12',
    'balanced_accuracy_12', 'matched_accuracy_12',
    'sensitivity_macro_12', 'specificity_macro_12',
    'precision_macro_12', 'recall_macro_12',
    'hamming_loss_12', 'jaccard_12',
    'hier_violation_rate_12', 'label_cooc_consistency_12',
    # loss
    'test_loss',
]


def save_experiment(
    config:   dict,
    metrics:  dict,
    csv_path: str = 'hmcn_ablation_results.csv',
) -> None:
    """
    Merge config + metrics into one row and append to the shared CSV.
    Creates the file with header if it doesn't exist yet.
    Missing columns are filled with NaN so the schema stays consistent
    across different ablation experiments.

    config must include at minimum:
        experiment, param_name, param_value,
        global_dim, local_dim, dropout, lr, weight_decay,
        lambda_viol, beta, batch_size, seed,
        best_epoch, train_loss_at_best, val_meta_roc_auc
    """
    row = {**config, **metrics}
    for col in _CSV_COLUMNS:
        if col not in row:
            row[col] = float('nan')

    df_new = pd.DataFrame([row])[_CSV_COLUMNS]

    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode='w', header=True,  index=False)

    print(f'  → Saved to {csv_path}  '
          f'[{config.get("experiment")} | '
          f'{config.get("param_name")}={config.get("param_value")}]')
