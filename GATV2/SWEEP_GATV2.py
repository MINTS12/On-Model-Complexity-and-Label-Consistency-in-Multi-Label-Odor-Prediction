import os
import random
import pickle
import csv
import copy
import itertools
import warnings
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_add_pool

from rdkit import Chem
from rdkit.Chem import rdPartialCharges, MolFromSmarts
from rdkit.Chem.rdchem import HybridizationType, ChiralType, BondStereo, BondType

from skmultilearn.model_selection import IterativeStratification

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    accuracy_score, jaccard_score, hamming_loss, matthews_corrcoef,
)

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

SCRIPT_START_TIME = datetime.now()
print(f"Script started at: {SCRIPT_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")

NUM_WORKERS = 0 if os.name == "nt" else 4
PERSISTENT_WORKERS = NUM_WORKERS > 0

# Format:
# [conv1_out, conv2_out, conv3_out, conv4_out,
#  global_mlp1_out, global_mlp2_out,
#  local1_transition_out, local2_transition_out,
#  dropout]

dims_baseline = [15,  20,  27,  36,   96,  63,  48,  63,  0.47]
dims_A        = [32,  48,  64,  96,  128, 138,  64, 138,  0.47]
dims_B        = [64,  96, 128, 160,  256, 256, 128, 256,  0.50]
dims_C        = [64,  96, 128, 192,  256, 256, 128, 256,  0.55]

dims = dims_C

df = pd.read_csv('https://raw.githubusercontent.com/MINTS12/On-Model-Complexity-and-Label-Consistency-in-Multi-Label-Odor-Prediction/refs/heads/main/Dataset/Multi-Labelled_Smiles_Odors_dataset.csv')

df.fillna(0, inplace=True)

META_CATEGORIES = {
    "macro_floral": ["floral", "rose", "jasmin", "lily", "muguet", "violet", "hyacinth",
                     "geranium", "lavender", "orangeflower", "chamomile", "hawthorn"],
    "macro_fruity": ["fruity", "apple", "apricot", "banana", "berry", "cherry", "grape",
                     "grapefruit", "lemon", "melon", "orange", "peach", "pear", "pineapple",
                     "plum", "raspberry", "strawberry", "tropical", "black currant", "fruit skin"],
    "macro_sweet": ["sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa",
                    "coconut", "creamy", "buttery", "milky", "dairy"],
    "macro_woody": ["woody", "cedar", "sandalwood", "pine", "vetiver", "terpenic",
                    "balsamic", "cortex"],
    "macro_green": ["green", "grassy", "herbal", "leafy", "hay", "tea", "fresh",
                    "cucumber", "vegetable", "weedy"],
    "macro_spicy": ["spicy", "cinnamon", "clove", "warm", "pungent", "sharp",
                    "cooling", "mint", "camphoreous"],
    "macro_animal_musk": ["animal", "musk", "leathery", "fishy", "sweaty", "meaty",
                          "beefy", "musty"],
    "macro_earthy": ["earthy", "mushroom", "nutty", "hazelnut", "roasted", "coffee",
                     "tobacco", "smoky", "popcorn"],
    "macro_citrus": ["citrus", "bergamot", "ozone", "clean", "soapy"],
    "macro_chemical": ["solvent", "ethereal", "metallic", "medicinal", "phenolic",
                       "sulfurous", "gassy", "burnt", "oily"],
    "macro_gourmand": ["almond", "malty", "rummy", "brandy", "cognac", "winey",
                       "cooked", "potato", "savory", "celery", "tomato", "radish",
                       "onion", "garlic", "cabbage", "cheesy"],
    "macro_powdery_amber": ["amber", "powdery", "anisic", "coumarinic", "orris",
                             "waxy", "aldehydic", "ketonic", "lactonic"],
}

label_columns = list(df.columns[2:])

child_parent_pairs = []

for parent_idx, (group_name, child_names) in enumerate(META_CATEGORIES.items()):
    for child_name in child_names:
        if child_name in label_columns:
            child_col_idx = label_columns.index(child_name)
            child_parent_pairs.append((child_col_idx, parent_idx))

_EN = {
    'H': 2.20, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
    'Se': 2.55, 'As': 2.18, 'Te': 2.10,
}

_HBD_SMARTS = MolFromSmarts('[#7,#8;!H0]')
_HBA_SMARTS = MolFromSmarts('[#7,#8]')


def smiles_to_graph(smiles: str, y_tensor: torch.Tensor) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    rdPartialCharges.ComputeGasteigerCharges(mol)

    donors    = {idx for match in mol.GetSubstructMatches(_HBD_SMARTS) for idx in match}
    acceptors = {idx for match in mol.GetSubstructMatches(_HBA_SMARTS) for idx in match}

    ring_info = mol.GetRingInfo()
    atom_ring_sizes: dict[int, set[int]] = {}
    for ring in ring_info.AtomRings():
        for idx in ring:
            atom_ring_sizes.setdefault(idx, set()).add(len(ring))

    node_features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        h   = atom.GetHybridization()
        c   = atom.GetChiralTag()
        rs  = atom_ring_sizes.get(idx, set())

        charge = atom.GetDoubleProp('_GasteigerCharge')
        if charge != charge:
            charge = 0.0

        node_features.append([
            atom.GetAtomicNum(),
            int(c == ChiralType.CHI_UNSPECIFIED),
            int(c == ChiralType.CHI_TETRAHEDRAL_CW),
            int(c == ChiralType.CHI_TETRAHEDRAL_CCW),
            int(c == ChiralType.CHI_OTHER),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            int(h == HybridizationType.SP),
            int(h == HybridizationType.SP2),
            int(h == HybridizationType.SP3),
            int(h not in (HybridizationType.SP,
                          HybridizationType.SP2,
                          HybridizationType.SP3)),
            int(atom.GetIsAromatic()),
            int(3 in rs),
            int(4 in rs),
            int(5 in rs),
            int(6 in rs),
            int(7 in rs),
            charge,
            int(idx in donors),
            int(idx in acceptors),
        ])

    src, dst = [], []
    edge_features = []

    for bond in mol.GetBonds():
        i  = bond.GetBeginAtomIdx()
        j  = bond.GetEndAtomIdx()
        bt = bond.GetBondType()

        en_i     = _EN.get(mol.GetAtomWithIdx(i).GetSymbol(), 2.55)
        en_j     = _EN.get(mol.GetAtomWithIdx(j).GetSymbol(), 2.55)
        en_delta = abs(en_i - en_j)

        feat = [
            int(bt == BondType.SINGLE),
            int(bt == BondType.DOUBLE),
            int(bt == BondType.TRIPLE),
            int(bt == BondType.AROMATIC),
            int(bond.IsInRing()),
            int(bond.GetStereo() != BondStereo.STEREONONE),
            int(bond.GetIsConjugated()),
            en_delta,
        ]

        src += [i, j]
        dst += [j, i]
        edge_features += [feat, feat]

    x          = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y_tensor, smiles=smiles)

df_graph = []
skipped  = []

for i in range(len(df)):
    smiles  = df['nonStereoSMILES'][i]
    y       = torch.tensor(df.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
    data    = smiles_to_graph(smiles, y)

    if data is None:
        skipped.append(i)
        continue

    df_graph.append(data)

if skipped:
    print(f"Warning: {len(skipped)} molecules skipped (unparseable SMILES): {skipped}")

print(f"Built {len(df_graph)} graphs")
print(f"Node feature dim : {df_graph[0].x.shape[1]}")
print(f"Edge feature dim : {df_graph[0].edge_attr.shape[1]}")

def create_stratified_splits(df, label_columns, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    np.random.seed(seed)
    X = np.arange(len(df)).reshape(-1, 1)
    y = label_columns

    holdout_ratio = val_ratio + test_ratio

    stratifier_1 = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[holdout_ratio, train_ratio],
    )

    train_idx, holdout_idx = next(stratifier_1.split(X, y))

    print(f"   -> Step 1 Complete: {len(train_idx)} Train samples, {len(holdout_idx)} Holdout samples.")

    relative_test_ratio = test_ratio / holdout_ratio
    relative_val_ratio = 1.0 - relative_test_ratio

    np.random.seed(seed)
    stratifier_2 = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[relative_test_ratio, relative_val_ratio],
    )

    X_holdout = X[holdout_idx]
    y_holdout = y[holdout_idx]

    val_idx_relative, test_idx_relative = next(stratifier_2.split(X_holdout, y_holdout))

    val_idx = holdout_idx[val_idx_relative]
    test_idx = holdout_idx[test_idx_relative]

    print(f"   -> Step 2 Complete: {len(val_idx)} Val samples, {len(test_idx)} Test samples.")

    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    df_test = df.iloc[test_idx].copy()

    return df_train, df_val, df_test, train_idx, val_idx, test_idx


SPLIT_CACHE = "fixed_split_indices.pkl"

if os.path.exists(SPLIT_CACHE):
    with open(SPLIT_CACHE, "rb") as f:
        train_idx, val_idx, test_idx = pickle.load(f)
    train_data = df.iloc[train_idx].copy()
    val_data   = df.iloc[val_idx].copy()
    test_data  = df.iloc[test_idx].copy()
    print(f"Loaded cached split from '{SPLIT_CACHE}' "
          f"({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test).")
else:
    train_data, val_data, test_data, train_idx, val_idx, test_idx = create_stratified_splits(
        df=df,
        label_columns=df.iloc[:, 2:].values,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=SEED,
    )
    with open(SPLIT_CACHE, "wb") as f:
        pickle.dump((train_idx, val_idx, test_idx), f)
    print(f"Computed a fresh split and cached it to '{SPLIT_CACHE}'.")

df_graph_train = []
df_graph_val = []
df_graph_test = []

for i in range(len(train_data)):
  smiles  = train_data['nonStereoSMILES'].iloc[i]
  y       = torch.tensor(train_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  data    = smiles_to_graph(smiles, y)

  if data is None:
      skipped.append(i)
      continue

  df_graph_train.append(data)

for i in range(len(val_data)):
  smiles  = val_data['nonStereoSMILES'].iloc[i]
  y       = torch.tensor(val_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  data    = smiles_to_graph(smiles, y)

  if data is None:
      skipped.append(i)
      continue

  df_graph_val.append(data)

for i in range(len(test_data)):
  smiles  = test_data['nonStereoSMILES'].iloc[i]
  y       = torch.tensor(test_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  data    = smiles_to_graph(smiles, y)

  if data is None:
      skipped.append(i)
      continue

  df_graph_test.append(data)

train_data = df_graph_train
val_data = df_graph_val
test_data = df_graph_test

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
val_loader   = DataLoader(val_data,   batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)

def calculate_all_metrics(loader, model, device, criterion, threshold=0.5):
    model.eval()
    total_loss = 0
    y_true_all = []
    y_probs_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            logits_global, logits_local1, logits_local2 = model(batch)

            beta = model.beta
            out = beta * logits_local2 + (1 - beta) * logits_global

            y = batch.y.view(batch.num_graphs, -1).float()
            loss = criterion(logits_global, logits_local1, logits_local2, y)

            total_loss += loss.item()

            probs = torch.sigmoid(out)

            preds = (probs > threshold).float()

            y_true_all.append(batch.y.view(batch.num_graphs, -1).float().cpu())
            y_probs_all.append(probs.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true = np.vstack(y_true_all)
    y_probs = np.vstack(y_probs_all)
    y_pred = np.vstack(y_pred_all)

    avg_loss = total_loss / len(loader)

    auroc = roc_auc_score(y_true, y_probs, average='micro')

    aucpr = average_precision_score(y_true, y_probs, average='micro')

    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return avg_loss, auroc, aucpr, precision, recall, f1, f1_macro

class SmellGATV2_HMCNF(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_heads = 1, num_classes=138, num_parents=12, beta=0.5, dims=dims):
        super(SmellGATV2_HMCNF, self).__init__()

        c1, c2, c3, c4, g1, g2, l1t, l2t, drop = dims

        self.beta = beta
        self.num_classes = num_classes
        self.num_parents = num_parents

        self.conv1 = GATv2Conv(num_node_features, c1, edge_dim=num_edge_features, heads=num_heads, concat=False)
        self.conv2 = GATv2Conv(c1, c2, edge_dim=num_edge_features, heads=num_heads, concat=False)
        self.conv3 = GATv2Conv(c2, c3, edge_dim=num_edge_features, heads=num_heads, concat=False)
        self.conv4 = GATv2Conv(c3, c4, edge_dim=num_edge_features, heads=num_heads, concat=False)

        self.mlp_input_dim = num_node_features + sum(dims[:4])

        self.global_mlp1 = Linear(self.mlp_input_dim, g1)
        self.global_bn1  = BatchNorm1d(g1)
        self.global_drop1 = Dropout(drop)

        self.global_mlp2 = Linear(g1, g2)
        self.global_bn2  = BatchNorm1d(g2)
        self.global_drop2 = Dropout(drop)

        self.global_out = Linear(g2, num_classes)

        self.local1_transition = Linear(g1, l1t)
        self.local1_bn         = BatchNorm1d(l1t)
        self.local1_out        = Linear(l1t, num_parents)

        self.local2_transition = Linear(g2, l2t)
        self.local2_bn         = BatchNorm1d(l2t)
        self.local2_out        = Linear(l2t, num_classes)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x0 = x
        x1 = F.selu(self.conv1(x0, edge_index, edge_attr=edge_attr))
        x2 = F.selu(self.conv2(x1, edge_index, edge_attr=edge_attr))
        x3 = F.selu(self.conv3(x2, edge_index, edge_attr=edge_attr))
        x4 = F.selu(self.conv4(x3, edge_index, edge_attr=edge_attr))

        g0 = global_add_pool(x0, batch)
        g1 = global_add_pool(x1, batch)
        g2 = global_add_pool(x2, batch)
        g3 = global_add_pool(x3, batch)
        g4 = global_add_pool(x4, batch)

        graph_repr = torch.cat([g0, g1, g2, g3, g4], dim=1)

        h1 = self.global_mlp1(graph_repr)
        h1 = self.global_bn1(h1)
        h1 = F.relu(h1)
        h1 = self.global_drop1(h1)

        h2 = self.global_mlp2(h1)
        h2 = self.global_bn2(h2)
        h2 = F.relu(h2)
        h2 = self.global_drop2(h2)

        logits_global = self.global_out(h2)

        l1 = F.relu(self.local1_bn(self.local1_transition(h1)))
        logits_local1 = self.local1_out(l1)

        l2 = F.relu(self.local2_bn(self.local2_transition(h2)))
        logits_local2 = self.local2_out(l2)

        return logits_global, logits_local1, logits_local2

def get_parent_labels(y_138, child_parent_pairs, num_parents=12):
    batch_size = y_138.shape[0]

    parent_tensors = [[] for _ in range(num_parents)]
    for child_idx, parent_idx in child_parent_pairs:
        parent_tensors[parent_idx].append(y_138[:, child_idx])

    cols = []
    for group in parent_tensors:
        if group:
            cols.append(torch.stack(group, dim=0).max(dim=0).values)
        else:
            cols.append(torch.zeros(batch_size, device=y_138.device))

    return torch.stack(cols, dim=1)

child_idxs  = [cp[0] for cp in child_parent_pairs]
parent_idxs = [cp[1] for cp in child_parent_pairs]

def hmcnf_loss(logits_global, logits_local1, logits_local2, child_idxs, parent_idxs,
               y_138, child_parent_pairs,
               pos_weight=None, lambda_hier=0.3):

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    y_parents = get_parent_labels(y_138, child_parent_pairs, num_parents=12)

    loss_global = bce(logits_global, y_138)

    bce_unweighted = torch.nn.BCEWithLogitsLoss()
    loss_local1 = bce_unweighted(logits_local1, y_parents)

    loss_local2 = bce(logits_local2, y_138)

    probs_global = torch.sigmoid(logits_global)
    probs_parent = get_parent_labels(probs_global, child_parent_pairs, num_parents=12)

    p_children = probs_global[:, child_idxs]
    p_parents  = probs_parent[:, parent_idxs]
    penalty    = (torch.relu(p_children - p_parents) ** 2).mean()

    penalty = penalty / len(child_parent_pairs)

    return loss_global + loss_local1 + loss_local2 + lambda_hier * penalty

def find_per_label_thresholds(val_loader, model, device, num_classes=138):
    model.eval()
    y_true_all = []
    y_probs_all = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            logits_global, logits_local1, logits_local2 = model(batch)
            beta = model.beta
            out = beta * logits_local2 + (1 - beta) * logits_global

            probs = torch.sigmoid(out)

            y_true_all.append(batch.y.view(batch.num_graphs, -1).float().cpu())
            y_probs_all.append(probs.cpu().numpy())

    y_true = np.vstack(y_true_all)
    y_probs = np.vstack(y_probs_all)

    best_thresholds = np.full(num_classes, 0.5)

    print(f"Sweeping thresholds for all {num_classes} labels individually...")

    for class_idx in range(num_classes):
        y_true_class = y_true[:, class_idx]
        y_probs_class = y_probs[:, class_idx]

        if np.sum(y_true_class) == 0:
            continue

        best_f1 = 0.0
        best_thresh = 0.5

        for thresh in np.arange(0.01, 1.0, 0.01):
            y_pred_class = (y_probs_class >= thresh).astype(int)
            current_f1 = f1_score(y_true_class, y_pred_class, average = 'macro', zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = thresh

        best_thresholds[class_idx] = best_thresh

    print("Done! Found 138 optimal thresholds.")
    return best_thresholds

def find_per_label_thresholds_12(val_loader, model, device, num_parents=12):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits_global, logits_local1, logits_local2 = model(batch)

            probs_12 = torch.sigmoid(logits_local1)

            y_138 = batch.y.view(batch.num_graphs, -1).float()
            y_12  = get_parent_labels(y_138, child_parent_pairs, num_parents=12)

            all_probs.append(probs_12.cpu())
            all_labels.append(y_12.cpu())

    all_probs  = torch.cat(all_probs,  dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    thresholds_12 = []
    for i in range(num_parents):
        best_thresh, best_f1 = 0.5, 0.0
        for t in torch.arange(0.1, 0.9, 0.01):
            preds = (all_probs[:, i] >= t.item()).astype(int)
            f1 = f1_score(all_labels[:, i], preds, average = 'macro', zero_division=0)
            if f1 > best_f1:
                best_f1    = f1
                best_thresh = t.item()
        thresholds_12.append(best_thresh)

    return torch.tensor(thresholds_12)

def run_sweep(
    train_data, val_data,
    df_graph_train,
    child_parent_pairs, child_idxs, parent_idxs,
    num_node_features, num_edge_features,
    lambda_values     = [0.0, 0.01, 0.05, 0.1],
    heads_values      = [1, 4, 8],
    t0_values         = [50, 100, 200],
    epochs_values     = [500, 1000],
    beta              = 0.5,
    dropout           = 0.47,
    learning_rate     = 1e-3,
    batch_size        = 128,
    num_classes       = 138,
    num_parents       = 12,
    results_csv       = "sweep_results.csv",
    checkpoint_dir    = "checkpoints",
    monitor           = "val_auroc",
    device            = None,
):
    gpu = '0'

    if device is None:
        device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print(f"Sweep device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    all_y = torch.stack([d.y for d in train_data])
    num_pos = all_y.sum(0)
    num_neg = len(train_data) - num_pos
    pos_weight = (num_neg / (num_pos + 1e-5)).to(device)

    set_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)

    fieldnames = [
        "run_id", "timestamp",
        "lambda_hier", "num_heads", "T_0", "epochs",
        "beta", "dropout", "lr",
        "best_epoch", "monitor_metric",
        "val_loss", "val_auroc", "val_aucpr",
        "val_precision", "val_recall", "val_f1_micro", "val_f1_macro",
    ]
    write_header = not os.path.exists(results_csv)
    csv_file = open(results_csv, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    def make_criterion(lambda_hier_val):
        return lambda lg, ll1, ll2, y: hmcnf_loss(
            lg, ll1, ll2, child_idxs, parent_idxs, y, child_parent_pairs,
            pos_weight=pos_weight, lambda_hier=lambda_hier_val,
        )

    def is_better(new_val, best_val):
        if monitor == "val_loss":
            return new_val < best_val
        return new_val > best_val

    best_init = float("inf") if monitor == "val_loss" else 0.0

    grid = list(itertools.product(lambda_values, heads_values, t0_values, epochs_values))
    print(f"\nTotal runs: {len(grid)}\n{'='*60}")

    for run_idx, (lam, heads, t0, epochs) in enumerate(grid):
        run_id  = f"run{run_idx:03d}_lam{lam}_h{heads}_t0{t0}_ep{epochs}"
        ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{run_idx+1}/{len(grid)}] {run_id}  started {ts_start}")

        set_seed(SEED)

        model = SmellGATV2_HMCNF(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_heads=heads,
            num_classes=num_classes,
            num_parents=num_parents,
            beta=beta,
        ).to(device)

        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=1)
        criterion = make_criterion(lam)

        best_metric   = best_init
        best_state    = None
        best_epoch    = 0
        history_auroc = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                lg, ll1, ll2 = model(batch)
                y = batch.y.view(batch.num_graphs, -1).float()
                loss = criterion(lg, ll1, ll2, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            val_loss, val_auroc, val_aucpr, val_prec, val_rec, val_f1, val_f1_macro = \
                calculate_all_metrics(val_loader, model, device, criterion)

            monitor_val = {
                "val_auroc":    val_auroc,
                "val_loss":     val_loss,
                "val_f1_macro": val_f1_macro,
                "val_aucpr":    val_aucpr,
            }[monitor]

            if is_better(monitor_val, best_metric):
                best_metric = monitor_val
                best_state  = copy.deepcopy(model.state_dict())
                best_epoch  = epoch + 1
                best_metrics_snapshot = dict(
                    val_loss=val_loss, val_auroc=val_auroc, val_aucpr=val_aucpr,
                    val_precision=val_prec, val_recall=val_rec,
                    val_f1_micro=val_f1, val_f1_macro=val_f1_macro,
                )

            if (epoch + 1) % 5 == 0:
                print(f"  ep {epoch+1:4d} | loss {val_loss:.4f} | "
                      f"auroc {val_auroc:.4f} | aucpr {val_aucpr:.4f} | f1_macro {val_f1_macro:.4f}")

        ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_best.pt")
        torch.save(best_state, ckpt_path)

        row = dict(
            run_id=run_id, timestamp=ts_start,
            lambda_hier=lam, num_heads=heads, T_0=t0, epochs=epochs,
            beta=beta, dropout=dropout, lr=learning_rate,
            best_epoch=best_epoch, monitor_metric=monitor,
            **best_metrics_snapshot,
        )
        writer.writerow(row)
        csv_file.flush()

        print(f"  → Best epoch {best_epoch} | auroc {best_metrics_snapshot['val_auroc']:.4f} "
              f"| f1_macro {best_metrics_snapshot['val_f1_macro']:.4f} "
              f"| aucpr {best_metrics_snapshot['val_aucpr']:.4f}")
        print(f"  → Saved to {ckpt_path}")

    csv_file.close()
    print(f"\n{'='*60}\nSweep complete. Results saved to '{results_csv}'.")

def evaluate_sweep(
    sweep_csv          = "sweep_results.csv",
    checkpoint_dir     = "checkpoints",
    val_data           = None,
    test_data          = None,
    child_parent_pairs = None,
    num_node_features  = None,
    num_edge_features  = None,
    num_classes        = 138,
    num_parents        = 12,
    output_csv         = "sweep_test_results.csv",
    batch_size         = 128,
    device             = None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluate device: {device}")

    val_loader  = DataLoader(val_data,  batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)

    child_idxs  = [cp[0] for cp in child_parent_pairs]
    parent_idxs = [cp[1] for cp in child_parent_pairs]

    hp_cols  = ["run_id", "timestamp",
                "lambda_hier", "num_heads", "T_0", "epochs",
                "beta", "dropout", "lr",
                "best_epoch", "monitor_metric"]

    val_cols = ["val_loss", "val_auroc", "val_aucpr",
                "val_precision", "val_recall", "val_f1_micro", "val_f1_macro"]

    test_cols_138 = [
        "test_auroc_138", "test_aucpr_138",
        "test_f1_macro_138", "test_f1_micro_138",
        "test_hier_violation_rate_138",
        "test_precision_micro_138", "test_recall_micro_138",
        "test_precision_macro_138", "test_recall_macro_138",
        "test_accuracy_138",
        "test_jaccard_macro_138", "test_jaccard_micro_138",
        "test_hamming_loss_138",
    ]
    test_cols_12 = [
        "test_auroc_12", "test_aucpr_12",
        "test_f1_macro_12", "test_f1_micro_12", "test_f1_instance_12",
        "test_balanced_accuracy_12",
        "test_accuracy_12",
        "test_sensitivity_macro_12", "test_specificity_macro_12",
        "test_hier_violation_rate_12",
        "test_label_cooc_consistency_12",
        "test_jaccard_macro_12", "test_jaccard_micro_12",
        "test_hamming_loss_12",
    ]
    misc_cols = ["test_loss"]

    fieldnames = hp_cols + val_cols + test_cols_138 + test_cols_12 + misc_cols

    write_header = not os.path.exists(output_csv)
    out_file = open(output_csv, "a", newline="")
    writer   = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        out_file.flush()

    with open(sweep_csv, "r") as f:
        sweep_rows = list(csv.DictReader(f))
    print(f"Found {len(sweep_rows)} runs in '{sweep_csv}'.\n{'='*60}")

    for i, row in enumerate(sweep_rows):
        run_id = row["run_id"]
        print(f"\n[{i+1}/{len(sweep_rows)}] {run_id}")

        num_heads = int(row["num_heads"])
        beta      = float(row["beta"])

        model = SmellGATV2_HMCNF(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_heads=num_heads,
            num_classes=num_classes,
            num_parents=num_parents,
            beta=beta,
        ).to(device)

        ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found at {ckpt_path}, skipping.")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        lambda_hier      = float(row["lambda_hier"])
        pos_weight_dummy = torch.ones(num_classes, device=device)
        criterion = lambda lg, ll1, ll2, y: hmcnf_loss(
            lg, ll1, ll2, child_idxs, parent_idxs, y, child_parent_pairs,
            pos_weight=pos_weight_dummy, lambda_hier=lambda_hier,
        )

        thresholds_138 = torch.tensor(
            find_per_label_thresholds(val_loader, model, device, num_classes=num_classes),
            dtype=torch.float32,
        ).to(device)
        thresholds_12 = find_per_label_thresholds_12(
            val_loader, model, device, num_parents=num_parents
        ).to(device)

        model.eval()
        y_true_138_all, y_probs_138_all = [], []
        y_true_12_all,  y_probs_12_all  = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits_global, logits_local1, logits_local2 = model(batch)

                out_138 = beta * logits_local2 + (1 - beta) * logits_global
                probs_138 = torch.sigmoid(out_138)

                probs_12 = torch.sigmoid(logits_local1)

                y_138 = batch.y.view(batch.num_graphs, -1).float()
                y_12  = get_parent_labels(y_138, child_parent_pairs, num_parents=num_parents)

                loss = criterion(logits_global, logits_local1, logits_local2, y_138)
                total_loss += loss.item()

                y_true_138_all.append(y_138.cpu())
                y_probs_138_all.append(probs_138.cpu().numpy())
                y_true_12_all.append(y_12.cpu())
                y_probs_12_all.append(probs_12.cpu().numpy())

        y_true_138 = np.vstack(y_true_138_all)
        y_probs_138 = np.vstack(y_probs_138_all)
        y_true_12  = np.vstack(y_true_12_all)
        y_probs_12  = np.vstack(y_probs_12_all)

        y_pred_138 = (y_probs_138 >= thresholds_138.cpu().numpy()).astype(int)
        y_pred_12  = (y_probs_12  >= thresholds_12.cpu().numpy()).astype(int)

        avg_loss = total_loss / len(test_loader)

        valid_cols_138 = [j for j in range(num_classes)
                          if len(np.unique(y_true_138[:, j])) > 1]

        auroc_138 = roc_auc_score(
            y_true_138[:, valid_cols_138], y_probs_138[:, valid_cols_138], average='macro')
        aucpr_138 = average_precision_score(
            y_true_138[:, valid_cols_138], y_probs_138[:, valid_cols_138], average='macro')

        violations, total_child_pos = 0, 0
        for child_idx, parent_idx in child_parent_pairs:
            child_pred  = y_pred_138[:, child_idx]
            parent_pred = y_pred_12[:, parent_idx]
            violations      += ((child_pred == 1) & (parent_pred == 0)).sum()
            total_child_pos += (child_pred == 1).sum()
        hier_violation_rate = violations / (total_child_pos + 1e-8)

        valid_cols_12 = [j for j in range(num_parents)
                         if len(np.unique(y_true_12[:, j])) > 1]

        auroc_12 = roc_auc_score(
            y_true_12[:, valid_cols_12], y_probs_12[:, valid_cols_12], average='macro')
        aucpr_12 = average_precision_score(
            y_true_12[:, valid_cols_12], y_probs_12[:, valid_cols_12], average='macro')

        bal_acc_scores, sens_scores, spec_scores = [], [], []
        for j in range(num_parents):
            if len(np.unique(y_true_12[:, j])) < 2:
                continue
            tn = ((y_pred_12[:, j] == 0) & (y_true_12[:, j] == 0)).sum()
            fp = ((y_pred_12[:, j] == 1) & (y_true_12[:, j] == 0)).sum()
            fn = ((y_pred_12[:, j] == 0) & (y_true_12[:, j] == 1)).sum()
            tp = ((y_pred_12[:, j] == 1) & (y_true_12[:, j] == 1)).sum()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            sens_scores.append(sens)
            spec_scores.append(spec)
            bal_acc_scores.append((sens + spec) / 2)

        bal_acc_12  = np.mean(bal_acc_scores)
        sensitivity = np.mean(sens_scores)
        specificity = np.mean(spec_scores)

        cooc_true = (y_true_12.T @ y_true_12)
        cooc_pred = (y_pred_12.T @ y_pred_12)
        cooc_true_norm = cooc_true / (cooc_true.max() + 1e-8)
        cooc_pred_norm = cooc_pred / (cooc_pred.max() + 1e-8)
        label_cooc_consistency = np.mean(np.abs(cooc_true_norm - cooc_pred_norm))

        out_row = {col: row.get(col, "") for col in hp_cols + val_cols}
        out_row.update({
            "test_auroc_138":               f"{auroc_138:.4f}",
            "test_aucpr_138":               f"{aucpr_138:.4f}",
            "test_f1_macro_138":            f"{f1_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}",
            "test_f1_micro_138":            f"{f1_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}",
            "test_hier_violation_rate_138": f"{hier_violation_rate:.4f}",
            "test_precision_micro_138":     f"{precision_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}",
            "test_recall_micro_138":        f"{recall_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}",
            "test_precision_macro_138":     f"{precision_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}",
            "test_recall_macro_138":        f"{recall_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}",
            "test_accuracy_138":            f"{accuracy_score(y_true_138, y_pred_138):.4f}",
            "test_jaccard_macro_138":       f"{jaccard_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}",
            "test_jaccard_micro_138":       f"{jaccard_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}",
            "test_hamming_loss_138":        f"{hamming_loss(y_true_138, y_pred_138):.4f}",
            "test_auroc_12":                f"{auroc_12:.4f}",
            "test_aucpr_12":                f"{aucpr_12:.4f}",
            "test_f1_macro_12":             f"{f1_score(y_true_12, y_pred_12, average='macro', zero_division=0):.4f}",
            "test_f1_micro_12":             f"{f1_score(y_true_12, y_pred_12, average='micro', zero_division=0):.4f}",
            "test_f1_instance_12":          f"{f1_score(y_true_12, y_pred_12, average='samples', zero_division=0):.4f}",
            "test_balanced_accuracy_12":    f"{bal_acc_12:.4f}",
            "test_accuracy_12":             f"{accuracy_score(y_true_12, y_pred_12):.4f}",
            "test_sensitivity_macro_12":    f"{sensitivity:.4f}",
            "test_specificity_macro_12":    f"{specificity:.4f}",
            "test_hier_violation_rate_12":  f"{hier_violation_rate:.4f}",
            "test_label_cooc_consistency_12": f"{label_cooc_consistency:.4f}",
            "test_jaccard_macro_12":        f"{jaccard_score(y_true_12, y_pred_12, average='macro', zero_division=0):.4f}",
            "test_jaccard_micro_12":        f"{jaccard_score(y_true_12, y_pred_12, average='micro', zero_division=0):.4f}",
            "test_hamming_loss_12":         f"{hamming_loss(y_true_12, y_pred_12):.4f}",
            "test_loss":                    f"{avg_loss:.4f}",
        })
        writer.writerow(out_row)
        out_file.flush()

        w = 62
        print(f"\n  {'─'*w}")
        print(f"  HMCN Fine — 138 labels  ({len(valid_cols_138)}/138 used for AUC)")
        print(f"  {'─'*w}")
        print(f"  {'ROC AUC':<32}: {auroc_138:.4f}")
        print(f"  {'PR AUC':<32}: {aucpr_138:.4f}")
        print(f"  {'F1 (macro)':<32}: {out_row['test_f1_macro_138']}")
        print(f"  {'F1 (micro)':<32}: {out_row['test_f1_micro_138']}")
        print(f"  {'Hier Violation Rate':<32}: {hier_violation_rate:.4f}")
        print(f"  {'Precision (micro)':<32}: {out_row['test_precision_micro_138']}")
        print(f"  {'Recall (micro)':<32}: {out_row['test_recall_micro_138']}")
        print(f"  {'Precision (macro)':<32}: {out_row['test_precision_macro_138']}")
        print(f"  {'Recall (macro)':<32}: {out_row['test_recall_macro_138']}")
        print(f"  {'Accuracy':<32}: {out_row['test_accuracy_138']}")
        print(f"  {'Jaccard (macro)':<32}: {out_row['test_jaccard_macro_138']}")
        print(f"  {'Jaccard (micro)':<32}: {out_row['test_jaccard_micro_138']}")
        print(f"  {'Hamming Loss':<32}: {out_row['test_hamming_loss_138']}")
        print(f"  {'─'*w}")
        print(f"  HMCN Meta — 12 groups   ({len(valid_cols_12)}/12 used for AUC)")
        print(f"  {'─'*w}")
        print(f"  {'ROC AUC':<32}: {auroc_12:.4f}")
        print(f"  {'PR AUC':<32}: {aucpr_12:.4f}")
        print(f"  {'F1 (macro)':<32}: {out_row['test_f1_macro_12']}")
        print(f"  {'F1 (micro)':<32}: {out_row['test_f1_micro_12']}")
        print(f"  {'Instance-F1':<32}: {out_row['test_f1_instance_12']}")
        print(f"  {'Balanced Accuracy':<32}: {bal_acc_12:.4f}")
        print(f"  {'Accuracy':<32}: {out_row['test_accuracy_12']}")
        print(f"  {'Sensitivity (macro)':<32}: {sensitivity:.4f}")
        print(f"  {'Specificity (macro)':<32}: {specificity:.4f}")
        print(f"  {'Hier Violation Rate':<32}: {hier_violation_rate:.4f}")
        print(f"  {'Label Co-occ Consistency':<32}: {label_cooc_consistency:.4f}")
        print(f"  {'Jaccard (macro)':<32}: {out_row['test_jaccard_macro_12']}")
        print(f"  {'Jaccard (micro)':<32}: {out_row['test_jaccard_micro_12']}")
        print(f"  {'Hamming Loss':<32}: {out_row['test_hamming_loss_12']}")
        print(f"  {'─'*w}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  {'─'*w}\n")

    out_file.close()
    print(f"\n{'='*60}\nDone. Test results saved to '{output_csv}'.")

def build_graphs(df_split, label_start_col=2):
    graphs = []
    n_skipped = 0
    for i in range(len(df_split)):
        smiles = df_split['nonStereoSMILES'].iloc[i]
        y = torch.tensor(df_split.iloc[i, label_start_col:].to_numpy(dtype=float), dtype=torch.float)
        data = smiles_to_graph(smiles, y)
        if data is None:
            n_skipped += 1
            continue
        graphs.append(data)
    if n_skipped:
        print(f"    (skipped {n_skipped} unparseable SMILES in this split)")
    return graphs


def compute_per_label_metrics_table(y_true, y_probs, y_pred, label_names):
    n_labels = y_true.shape[1]
    rows = []
    for j in range(n_labels):
        yt = y_true[:, j].astype(int)
        pr = y_probs[:, j]
        yp = y_pred[:, j].astype(int)

        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())

        precision   = precision_score(yt, yp, zero_division=0)
        sensitivity = recall_score(yt, yp, zero_division=0)
        specificity = recall_score(yt, yp, pos_label=0, zero_division=0)
        f1          = f1_score(yt, yp, zero_division=0)
        bal_acc     = (sensitivity + specificity) / 2

        if len(np.unique(yt)) > 1:
            roc_auc = roc_auc_score(yt, pr)
            pr_auc  = average_precision_score(yt, pr)
        else:
            roc_auc = np.nan
            pr_auc  = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mcc = matthews_corrcoef(yt, yp)

        rows.append({
            "label": label_names[j], "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Bal.Acc": bal_acc, "MCC": mcc, "F1": f1,
            "ROC_AUC": roc_auc, "PR_AUC": pr_auc,
            "Precision": precision, "Sensitivity": sensitivity, "Specificity": specificity,
        })
    return pd.DataFrame(rows)


def select_best_config(sweep_df, monitor):
    ascending = (monitor == "val_loss")
    return sweep_df.sort_values(monitor, ascending=ascending).iloc[0]


def evaluate_best_model_per_label(
    best_row, checkpoint_dir, val_data, test_data, label_names,
    num_node_features, num_edge_features,
    num_classes=138, num_parents=12, batch_size=128, device=None,
):
    """Reloads the best checkpoint, recalibrates per-label thresholds on val, evaluates on test."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader  = DataLoader(val_data,  batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)

    run_id    = best_row["run_id"]
    num_heads = int(best_row["num_heads"])
    beta      = float(best_row["beta"])

    model = SmellGATV2_HMCNF(
        num_node_features=num_node_features, num_edge_features=num_edge_features,
        num_heads=num_heads, num_classes=num_classes, num_parents=num_parents, beta=beta,
    ).to(device)

    ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f"  [best-model per-label] {run_id} -> calibrating thresholds on val set...")
    thresholds_138 = torch.tensor(
        find_per_label_thresholds(val_loader, model, device, num_classes=num_classes),
        dtype=torch.float32,
    ).to(device)

    y_true_all, y_probs_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits_global, _, logits_local2 = model(batch)
            out   = beta * logits_local2 + (1 - beta) * logits_global
            probs = torch.sigmoid(out)
            y_true_all.append(batch.y.view(batch.num_graphs, -1).float().cpu())
            y_probs_all.append(probs.cpu().numpy())

    y_true  = np.vstack(y_true_all)
    y_probs = np.vstack(y_probs_all)
    y_pred  = (y_probs >= thresholds_138.cpu().numpy()).astype(int)

    table = compute_per_label_metrics_table(y_true, y_probs, y_pred, label_names)
    table.insert(0, "run_id",      run_id)
    table.insert(1, "lambda_hier", best_row["lambda_hier"])
    table.insert(2, "num_heads",   best_row["num_heads"])
    table.insert(3, "T_0",         best_row["T_0"])
    table.insert(4, "epochs",      best_row["epochs"])
    return table


def run_single_split_sweep(
    df,
    seed,
    train_ratio             = 0.8,
    val_ratio                = 0.1,
    test_ratio               = 0.1,
    lambda_values            = [0.0, 0.01, 0.05, 0.1, 0.3],
    heads_values             = [1, 4, 8],
    t0_values                = [50],
    epochs_values            = [500, 1000],
    beta                     = 0.5,
    dropout                  = 0.47,
    learning_rate            = 1e-3,
    batch_size               = 128,
    num_classes              = 138,
    num_parents              = 12,
    monitor                  = "val_auroc",
    results_dir              = "single_split_results",
):
    os.makedirs(results_dir, exist_ok=True)
    label_columns = df.iloc[:, 2:].values

    df_train, df_val, df_test, train_idx, val_idx, test_idx = create_stratified_splits(
        df=df,
        label_columns=label_columns,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    print("  Building graphs...")
    train_graphs = build_graphs(df_train)
    val_graphs   = build_graphs(df_val)
    test_graphs  = build_graphs(df_test)

    results_csv = os.path.join(results_dir, "sweep_results.csv")
    ckpt_dir    = os.path.join(results_dir, "checkpoints")
    test_csv    = os.path.join(results_dir, "test_results.csv")

    run_sweep(
        train_data          = train_graphs,
        val_data             = val_graphs,
        df_graph_train       = train_graphs,
        child_parent_pairs   = child_parent_pairs,
        child_idxs           = child_idxs,
        parent_idxs          = parent_idxs,
        num_node_features    = train_graphs[0].x.shape[1],
        num_edge_features    = train_graphs[0].edge_attr.shape[1],
        lambda_values        = lambda_values,
        heads_values         = heads_values,
        t0_values             = t0_values,
        epochs_values        = epochs_values,
        beta                  = beta,
        dropout               = dropout,
        learning_rate        = learning_rate,
        batch_size            = batch_size,
        num_classes           = num_classes,
        num_parents           = num_parents,
        results_csv           = results_csv,
        checkpoint_dir        = ckpt_dir,
        monitor               = monitor,
    )

    evaluate_sweep(
        sweep_csv            = results_csv,
        checkpoint_dir        = ckpt_dir,
        val_data              = val_graphs,
        test_data             = test_graphs,
        child_parent_pairs    = child_parent_pairs,
        num_node_features     = train_graphs[0].x.shape[1],
        num_edge_features     = train_graphs[0].edge_attr.shape[1],
        num_classes           = num_classes,
        num_parents           = num_parents,
        output_csv            = test_csv,
        batch_size            = batch_size,
    )

    test_df = pd.read_csv(test_csv)
    sweep_df = pd.read_csv(results_csv)
    best_row = select_best_config(sweep_df, monitor)
    print(f"  Best config ({monitor} = {best_row[monitor]}): {best_row['run_id']}")

    per_label_df = evaluate_best_model_per_label(
        best_row           = best_row,
        checkpoint_dir      = ckpt_dir,
        val_data            = val_graphs,
        test_data           = test_graphs,
        label_names         = list(df.columns[2:]),
        num_node_features   = train_graphs[0].x.shape[1],
        num_edge_features   = train_graphs[0].edge_attr.shape[1],
        num_classes         = num_classes,
        num_parents         = num_parents,
        batch_size          = batch_size,
    )

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"  Deleted checkpoint in '{ckpt_dir}' to free disk space.")

    return test_df, per_label_df


seed_rng = random.Random()

N_RUNS = 50
master_csv    = "random_seed_sweep_results_GATV2.csv"
per_label_csv = "best_model_per_label_test_metrics_GATV2.csv"

for i in range(N_RUNS):
    s = seed_rng.randint(0, 2**31 - 1)
    SEED = s
    set_seed(SEED)

    print(f"\n########## RUN {i+1}/{N_RUNS} — seed={SEED} ##########")

    test_df, per_label_df = run_single_split_sweep(
        df=df,
        seed=SEED,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        lambda_values=[0, 0.01, 0.05, 0.1, 0.3],
        heads_values=[2, 4, 8],
        t0_values=[50],
        epochs_values=[500, 1000],
        dropout=0.55,
        monitor="val_aucpr",
        results_dir=f"single_split_results_GATV2_seed{SEED}",
    )

    test_df.insert(0, "seed", SEED)
    write_header = not os.path.exists(master_csv)
    test_df.to_csv(master_csv, mode="a", header=write_header, index=False)

    per_label_df.insert(0, "seed", SEED)
    write_header_pl = not os.path.exists(per_label_csv)
    per_label_df.to_csv(per_label_csv, mode="a", header=write_header_pl, index=False)


script_end_time = datetime.now()
elapsed = script_end_time - SCRIPT_START_TIME
print(f"\nScript finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total elapsed time: {elapsed}")