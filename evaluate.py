import os
import argparse
import math
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch_geometric

from pair_utils import collect_pairs

def _safe_div(n, d):
    return float(n) / float(d) if d else 0.0


def _mcc(tp, fp, tn, fn):
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _safe_div((tp * tn - fp * fn), denom) if denom else 0.0


def _confusion_from_labels(y_true, y_pred):
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return tp, fp, tn, fn


def _f1_macro(tp, fp, tn, fn):
    """Macro F1 across positive/negative classes from confusion counts."""
    f1_pos = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_neg = _safe_div(2 * tn, 2 * tn + fn + fp)
    return 0.5 * (f1_pos + f1_neg)


def build_loader(pairs, batch_size, workers):
    from GCN import GraphDataset_position as GraphDataset
    from GCN import GNNPolicy_position as GNNPolicy

    data = GraphDataset(pairs)
    loader = torch_geometric.loader.DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    return loader, GNNPolicy


def evaluate(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pairs = collect_pairs(args.test_root)
    if not pairs:
        raise RuntimeError('No .bg/.sol pairs found under --test-root')

    loader, GNNPolicy = build_loader(pairs, batch_size=1, workers=args.workers)
    model = GNNPolicy().to(device)
    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    total_selected_groups = 0

    y_true_sel_all: List[int] = []
    y_pred_sel_all: List[int] = []

    with torch.no_grad():
        for gi, batch in enumerate(loader):
            batch = batch.to(device)

            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10
            logits = model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            ).squeeze().detach().cpu()

            varInds = batch.varInds[0]
            varname_map = varInds[0][0].detach().cpu()
            alpha_idx = varInds[1][0].long().detach().cpu()
            group_ptr = varInds[2][0].long().detach().cpu()

            n_var = len(varname_map)
            logits_slice = logits[:n_var].squeeze()

            sols = batch.solutions.detach().cpu().reshape(-1)
            pair_bg, pair_sol = pairs[gi] if gi < len(pairs) else ('?', '?')
            if sols.numel() == 0:
                print(f"[WARN] empty solutions; skip this graph. bg={pair_bg} sol={pair_sol}")
                continue
            if varname_map.numel() == 0:
                print(f"[WARN] empty varname map; skip this graph. bg={pair_bg} sol={pair_sol}")
                continue
            max_idx = int(varname_map.max().item()) if varname_map.numel() > 0 else -1
            if sols.numel() <= max_idx:
                print(f"[WARN] solution vector too short (len={sols.numel()}) for max index {max_idx}; skip graph. bg={pair_bg} sol={pair_sol}")
                continue
            sols_graph = sols[varname_map]
            if alpha_idx.numel() == 0:
                print(f"[WARN] no alpha indices in this graph; skip. bg={pair_bg} sol={pair_sol}")
                continue
            sols_alpha = sols_graph[alpha_idx]

            n_groups = int(group_ptr.shape[0] - 1)

            group_infos = []  # (conf, g)
            groups_store = []  # per-group details for later selection/metrics
            for g in range(n_groups):
                lo = int(group_ptr[g].item())
                hi = int(group_ptr[g + 1].item())
                if hi <= lo:
                    continue
                group_logits = logits_slice[alpha_idx[lo:hi]]
                probs = F.softmax(group_logits, dim=0)
                p_sorted, idx_sorted = torch.sort(probs, descending=True)
                p1 = float(p_sorted[0].item())
                p2 = float(p_sorted[1].item()) if probs.numel() >= 2 else 0.0
                conf = p1 - p2
                pred_top_local = int(idx_sorted[0].item())

                gsol = sols_alpha[lo:hi]
                true_top_local = int(torch.argmax(gsol).item()) if gsol.numel() > 0 else 0

                group_infos.append((conf, g))

                y_true_group = [int(v) for v in gsol.detach().cpu().tolist()]
                y_pred_group = [1 if i == pred_top_local else 0 for i in range(len(y_true_group))]

                groups_store.append({
                    'pred_top_local': pred_top_local,
                    'true_top_local': true_top_local,
                    'conf': conf,
                    'y_true': y_true_group,
                    'y_pred': y_pred_group,
                })

            group_infos.sort(key=lambda x: x[0], reverse=True)
            k = int(math.ceil(max(0.0, min(100.0, args.p_top)) / 100.0 * len(group_infos)))
            selected = group_infos[:k]

            total_selected_groups += len(selected)
            y_true_sel: List[int] = []
            y_pred_sel: List[int] = []
            for _conf, _g in selected:
                details = groups_store[_g]
                y_true_sel.extend(details['y_true'])
                y_pred_sel.extend(details['y_pred'])

            y_true_sel_all.extend(y_true_sel)
            y_pred_sel_all.extend(y_pred_sel)

    if total_selected_groups > 0:
        tp, fp, tn, fn = _confusion_from_labels(y_true_sel_all, y_pred_sel_all)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = _f1_macro(tp, fp, tn, fn)
        bal_acc = (recall + specificity) / 2.0
        acc_bin = _safe_div(tp + tn, tp + tn + fp + fn)
        mcc = _mcc(tp, fp, tn, fn)
        print(f'count={len(y_pred_sel_all)} accuracy={acc_bin:.6f} precision={precision:.6f} recall={recall:.6f} f1_macro={f1:.6f} balanced_accuracy={bal_acc:.6f} matthews_correlation_coefficient={mcc:.6f}')


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate alpha predictions with hard group-argmax labels, optionally restricted to top-p% groups by confidence (p1-p2).')
    p.add_argument('--test-root', required=True, help='Test root (recursively contains .bg/.sol pairs)')
    p.add_argument('--model', required=True, help='Path to trained GNN .pth')
    p.add_argument('--p-top', type=float, default=100.0, help='Select top p%% groups by confidence (p1-p2) per graph')
    p.add_argument('--workers', type=int, default=0, help='Dataloader workers')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
