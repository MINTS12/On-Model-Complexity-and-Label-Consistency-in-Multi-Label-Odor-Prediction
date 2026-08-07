# Multi-Label Molecular Odor Prediction

Code for our work on predicting odor descriptors from molecular structure ,  comparing
simple tabular baselines (Random Forest, HMCN-F) against graph neural networks
(GCN, GATv2) to see whether the added complexity actually pays off at this dataset
scale. Covers 138 fine-grained odor labels grouped into 12 broader categories.

This repo is a work in progress ,  more code going in
over the next few days.

##Structure
- `tabular/` – features + Random Forest / HMCN-F models
- `graph/` – GCN and GATv2 training code
- `analysis/` – results and per-label consistency analysis
