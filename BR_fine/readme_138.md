# Odor Descriptor Hyperparameter Sweep (Binary Relevance)

`BR_fine138_sweep.py` runs a hyperparameter sweep to predict 138 fine-grained
odor descriptors from molecular features, using Binary Relevance over five
models: Logistic Regression, Random Forest, XGBoost, SVM, KNN.

## Requirements

Same folder as the script:
- `hmcn_dataset.csv` — features (MACCS, Morgan, Mordred) + `fine_*` labels
- `hmcn_eval.py` — shared evaluation utility

```bash
pip install numpy pandas scikit-learn xgboost scikit-multilearn iterative-stratification
```

## Running

**Debug mode (test setup first):** set `DEBUG = True` at line 37, then run:
```bash
python BR_fine138_sweep.py
```
Runs 1 fold, 1 model config, 6 labels. 

**Full run:** set `DEBUG = False`. 22 configs × 5 folds × 138 labels — run in
the background:
```bash
nohup python3 -u BR_fine138_sweep.py > /dev/null 2>&1 &
```
Progress is logged to `logs/`.

## What it does

1. **Split:** caches an 80/10/10 train/val/test split, then a 5-fold CV split
   inside the training subset. The sweep only touches this training subset.
2. **Feature cleaning (per fold):** drops zero-variance features, scales
   Mordred descriptors, drops Mordred features with Pearson correlation > 0.95.
3. **Selection metric:** macro PR-AUC
4. **Auto-resume:** re-running the script skips any (fold, model, config)
   combination already present in the results CSV.

## Outputs

- `sweep_splits/` — cached `.pkl` files: the reference train/val/test split
  and the 5-fold CV split, for reproducibility.
- `logs/` — timestamped run logs.
- `outputs_fine138_sweep/` — results. Key file:
  `baseline_fine138_sweep_winning_configs.csv` (best config per model, by
  macro PR-AUC). Other CSVs hold per-label/fold/model detail and aggregates.