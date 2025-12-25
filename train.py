import os
import argparse
import random
import time
import torch
import torch.nn.functional as F
import multiprocessing as mp
import torch_geometric

from pair_utils import collect_pairs


def save_loss_plot(train_hist, valid_hist, save_path):
    """Save train/valid loss curves to a PNG (copy of train.save_loss_plot)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(train_hist, label="train")
    plt.plot(valid_hist, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def build_loaders_from_pairs(
    train_pairs,
    valid_pairs,
    batch_size,
    num_workers,
    persistent=False,
    pin_memory=True,
    prefetch_factor=2,
):
    """Build DataLoaders and return (train_loader, valid_loader, GNNPolicy) (copy of train.build_loaders_from_pairs)."""
    from GCN import GraphDataset_position as GraphDataset
    from GCN import GNNPolicy_position as GNNPolicy

    train_data = GraphDataset(train_pairs)
    valid_data = GraphDataset(valid_pairs)

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

    try:
        if num_workers > 0:
            ctx = mp.get_context("spawn")
            train_loader = torch_geometric.loader.DataLoader(
                train_data,
                shuffle=True,
                multiprocessing_context=ctx,
                **dl_kwargs,
            )
            valid_loader = torch_geometric.loader.DataLoader(
                valid_data,
                shuffle=False,
                multiprocessing_context=ctx,
                **dl_kwargs,
            )
        else:
            train_loader = torch_geometric.loader.DataLoader(
                train_data,
                shuffle=True,
                **dl_kwargs,
            )
            valid_loader = torch_geometric.loader.DataLoader(
                valid_data,
                shuffle=False,
                **dl_kwargs,
            )
    except TypeError:
        dl_kwargs.pop("prefetch_factor", None)
        train_loader = torch_geometric.loader.DataLoader(
            train_data,
            shuffle=True,
            **dl_kwargs,
        )
        valid_loader = torch_geometric.loader.DataLoader(
            valid_data,
            shuffle=False,
            **dl_kwargs,
        )
    return train_loader, valid_loader, GNNPolicy


def train_one_epoch_gumbel(
    predict,
    data_loader,
    device,
    optimizer=None,
    weight_norm=100.0,
    alpha_loss_mode="gumbel",
    tau=1.0,
):
    """
    alpha_loss_mode:
      - 'gumbel': 组内 Gumbel-Softmax CE（用 group_ptr + alpha_idx）
      - 'BCE'   : per-alpha BCE
    """
    if optimizer is not None:
        predict.train()
    else:
        predict.eval()

    mean_loss = 0.0
    n_graphs = 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device, non_blocking=True)

            solInd = batch.nsols
            target_sols = []
            target_vals = []
            solEndInd = 0
            valEndInd = 0
            for i in range(solInd.shape[0]):
                nvar = len(batch.varInds[i][0][0])
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch.objVals[valStartInd:valEndInd]
                target_sols.append(sols)
                target_vals.append(vals)

            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10
            logits = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            probs = logits.sigmoid()

            loss = 0.0
            index_arrow = 0

            for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
                exp_weight = torch.exp(-vals / weight_norm)
                weight = exp_weight / exp_weight.sum()

                varInds = batch.varInds[ind]
                varname_map = varInds[0][0]
                alpha_idx = varInds[1][0].long()
                group_ptr = varInds[2][0].long() if len(varInds) > 2 else None

                sols_graph = sols[:, varname_map]
                sols_alpha = sols_graph[:, alpha_idx]

                n_var = batch.ntvars[ind]
                logits_slice = logits[index_arrow:index_arrow + n_var].squeeze()
                probs_slice = probs[index_arrow:index_arrow + n_var].squeeze()
                index_arrow += n_var

                if (
                    str(alpha_loss_mode) == "gumbel"
                    and group_ptr is not None
                    and group_ptr.numel() > 1
                ):
                    logits_alpha = logits_slice[alpha_idx]
                    graph_ce_sum = 0.0
                    group_count = 0
                    for g in range(group_ptr.shape[0] - 1):
                        lo = group_ptr[g].item()
                        hi = group_ptr[g + 1].item()
                        if hi <= lo:
                            continue
                        group_logits = logits_alpha[lo:hi]
                        if group_logits.numel() <= 1:
                            continue
                        group_count += 1

                        targets = torch.argmax(sols_alpha[:, lo:hi], dim=1)

                        group_probs = F.gumbel_softmax(
                            group_logits,
                            tau=tau,
                            hard=True,
                            dim=0,
                        )

                        p_true = group_probs[targets].clamp(min=1e-8)
                        ce = -torch.log(p_true)
                        graph_ce_sum = graph_ce_sum + (ce * weight).sum()

                    if group_count > 0:
                        loss = loss + (graph_ce_sum / float(group_count))
                    else:
                        loss = loss + 0.0
                else:
                    pre_sols = probs_slice[alpha_idx]
                    pos_loss = -(pre_sols + 1e-8).log()[None, :] * (sols_alpha == 1).float()
                    neg_loss = -(1 - pre_sols + 1e-8).log()[None, :] * (sols_alpha == 0).float()
                    sum_loss = pos_loss + neg_loss
                    sample_loss = sum_loss * weight[:, None]
                    loss = loss + sample_loss.sum()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += float(loss.item())
            n_graphs += batch.num_graphs

    mean_loss /= max(1, n_graphs)
    return mean_loss


def main():
    ap = argparse.ArgumentParser(
        description="Train alpha predictor with Gumbel-Softmax over alpha groups"
    )
    ap.add_argument("--train-root", required=True, help="Train root")
    ap.add_argument("--run-id", default=None, help="Optional run id for outputs (folder name under ./experiments/)")
    ap.add_argument("--loss", default="gumbel", choices=["gumbel", "BCE"], help="Loss type: 'gumbel' = group-level Gumbel-Softmax CE; 'BCE' = per-alpha BCE")
    ap.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--tau", type=float, default=1.0, help="Gumbel-Softmax temperature")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = args.run_id or "run_gumbel"
    exp_root = os.path.join("experiments", run_id)
    model_dir = os.path.join(exp_root, "model")
    log_dir = os.path.join(exp_root, "training")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "train.log")
    log_file = open(log_file_path, "wb")

    all_pairs = collect_pairs(args.train_root)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    valid_frac = 0.2
    cut = max(1, min(n - 1, int((1.0 - valid_frac) * n)))
    train_pairs = all_pairs[:cut]
    valid_pairs = all_pairs[cut:]
    print(f"Total: {n}  Train: {len(train_pairs)}  Valid: {len(valid_pairs)}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    persistent = args.workers > 0
    prefetch = 4 if args.workers > 0 else None
    train_loader, valid_loader, GNNPolicy = build_loaders_from_pairs(
        train_pairs,
        valid_pairs,
        args.batch_size,
        args.workers,
        persistent=persistent,
        pin_memory=True,
        prefetch_factor=prefetch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    model = GNNPolicy().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    train_hist, valid_hist = [], []
    loss_plot_path = os.path.join(log_dir, "loss_curve.png")

    for epoch in range(args.epochs):
        tic = time.time()
        tr = train_one_epoch_gumbel(
            model,
            train_loader,
            device,
            optimizer=optim,
            alpha_loss_mode=args.loss,
            tau=args.tau,
        )
        va = train_one_epoch_gumbel(
            model,
            valid_loader,
            device,
            optimizer=None,
            alpha_loss_mode=args.loss,
            tau=args.tau,
        )

        train_hist.append(tr)
        valid_hist.append(va)
        save_loss_plot(train_hist, valid_hist, loss_plot_path)

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), os.path.join(model_dir, "model_best.pth"))
        torch.save(model.state_dict(), os.path.join(model_dir, "model_last.pth"))

        msg = (
            f"@epoch{epoch}   "
            f"Train loss:{tr:.6f}   "
            f"Valid loss:{va:.6f}    "
            f"TIME:{time.time()-tic:.2f}\n"
        )
        print(msg.strip())
        log_file.write(msg.encode())
        log_file.flush()

    print("done")


if __name__ == "__main__":
    main()
