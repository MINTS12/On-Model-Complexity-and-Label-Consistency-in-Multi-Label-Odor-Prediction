# Odor Metacategory Hyperparameter Sweep (Binary Relevance)

`baseline_meta12_sweep.py` runs a hyperparameter sweep to predict **12 OR-derived metacategories**
odor descriptors from molecular features, using Binary Relevance over five
models: Logistic Regression, Random Forest, XGBoost, SVM, KNN.


## Requirements

Same folder as the script:
- `hmcn_dataset.csv`
- `hmcn_eval.py`

```bash
pip install numpy pandas scikit-learn xgboost scikit-multilearn iterative-stratification
```

## Running

**Note:** `DEBUG = True` by default in this file — set to `False` before a real run.

**Debug mode:**
```bash
python baseline_meta12_sweep.py
```
Runs 1 fold, 1 model config, 6 labels. Outputs go to `outputs_meta12_sweep_debug/`.

**Full run:**
```bash
nohup python3 -u baseline_meta12_sweep.py > /dev/null 2>&1 &
```
22 configs × 5 folds × 12 labels. Progress logged to `logs/`.


## Running

**Debug mode (test setup first):** set `DEBUG = True` , then run:
```bash
python baseline_meta12_sweep.py
```
Runs 1 fold, 1 model config, 6 labels. Outputs go to `outputs_meta12_sweep_debug/`.

**Full run:** set `DEBUG = False`. 22 configs × 5 folds × 138 labels — run in
the background:
```bash
nohup python3 -u BR_fine138_sweep.py > /dev/null 2>&1 &
```
Progress is logged to `logs/`

## What it does

1. **Split:** caches an 80/10/10 train/val/test split, then a 5-fold CV split
   inside the training subset. The sweep only touches this training subset.
2. **Feature cleaning (per fold):** drop zero-variance features, scale
   Mordred descriptors, drop Mordred features with correlation > 0.95.
3. **Selection metric:** macro PR-AUC, threshold-free.
4. **Auto-resume:** skips any (fold, model, config) already in the results CSV.

## Outputs

- `sweep_splits/` — cached `.pkl` split files: `reference_split_seed42.pkl`,
  `sweep_cv_folds_seed42.pkl`.
- `logs/` — timestamped run logs.
- `outputs_meta12_sweep/` — results. Key file:
  `baseline_meta12_sweep_winning_configs.csv` best config per model, by
  macro PR-AUC on the 12 metacategories
