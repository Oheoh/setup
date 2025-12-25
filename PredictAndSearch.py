import argparse
import math
import os
from pathlib import Path
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp

import gurobipy as gp
from gurobipy import GRB

from alpha_utils import group_alphas_sorted, build_c_lv_feature, build_t_feature
from helper import get_a_new2

def build_policy():
    """Return (model, device). Always uses positional encoding."""
    from GCN import GNNPolicy_position as GNNPolicy  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return GNNPolicy().to(device), device


def select_groups_by_confidence(logits, groups, top_percent):
    info = {}
    for key, idxs in groups.items():
        g_logits = logits[idxs]
        probs = F.softmax(g_logits, dim=0)
        p_sorted, idx_sorted = torch.sort(probs, descending=True)
        p1 = float(p_sorted[0].item())
        top1_local = int(idx_sorted[0].item())
        top1_idx = idxs[top1_local]
        p2 = float(p_sorted[1].item()) if probs.numel() >= 2 else 0.0
        conf = p1 - p2
        info[key] = (top1_idx, p1, p2, conf)

    all_keys = list(info.keys())
    all_keys.sort(key=lambda k: info[k][3], reverse=True)
    n_groups = len(all_keys)
    k = math.ceil(max(0.0, min(100.0, top_percent)) / 100.0 * n_groups)
    selected_keys = all_keys[:k]
    return selected_keys, info


def add_tr_constraints(m, var_map, x_star_names, delta):
    alphas = []
    for name, xi in x_star_names.items():
        xv = var_map.get(name)
        if xv is None:
            continue
        a = m.addVar(name=f"tr_{name}", vtype=GRB.CONTINUOUS, lb=0.0)
        alphas.append(a)
        m.addConstr(a >= xv - xi, name=f"tr_up_{name}")
        m.addConstr(a >= xi - xv, name=f"tr_dn_{name}")

    if alphas:
        m.addConstr(gp.quicksum(alphas) <= float(delta), name="tr_budget")

def solve_one(mps_path, policy, device, p_top, time_limit, threads, logdir):
    A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(mps_path)
    vnames = [name for name in v_map]
    alpha_idx = [i for i in b_vars if vnames[i].startswith("alpha_")]
    if not alpha_idx:
        raise RuntimeError("No alpha_* binaries found.")

    policy.eval()
    c_feat = c_nodes.to(torch.float32)
    from GCN import postion_get  # type: ignore
    v_feat = postion_get(v_nodes).to(torch.float32)
    c_lv = build_c_lv_feature(v_map).unsqueeze(1).to(torch.float32)
    t_feat = build_t_feature(v_map).unsqueeze(1).to(torch.float32)
    v_feat = torch.cat([v_feat, c_lv, t_feat], dim=1)
    edge_idx = A._indices().to(torch.long)
    edge_val = A._values().unsqueeze(1).to(torch.float32)

    with torch.no_grad():
        logits = policy(
            c_feat.to(device), edge_idx.to(device), edge_val.to(device), v_feat.to(device)
        ).squeeze().detach().cpu()

    groups = group_alphas_sorted(vnames, alpha_idx)
    selected_keys, info_map = select_groups_by_confidence(logits, groups, p_top)

    xstar_sel = {}
    for key in selected_keys:
        idxs = groups[key]
        top1_idx, p1, p2, conf = info_map[key]
        top1_name = vnames[top1_idx]
        for i in idxs:
            nm = vnames[i]
            xstar_sel[nm] = 1 if nm == top1_name else 0

    n_groups = len(groups)
    n_selected = len(selected_keys)
    delta = 2 * n_selected

    gp.setParam("LogToConsole", 0)
    m = gp.read(mps_path)
    m.Params.TimeLimit = float(time_limit)
    m.Params.Threads = int(threads)
    m.Params.MIPFocus = 1

    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    p = Path(mps_path)
    parent_tag = p.parent.name
    stem = p.stem
    base = f"{parent_tag}_{stem}" if parent_tag else stem
    m.Params.LogFile = str(logdir / f"{base}_ptop-{p_top:.1f}.log")

    mvars = m.getVars()
    var_map = {v.VarName: v for v in mvars}

    for name, val in xstar_sel.items():
        v = var_map.get(name)
        if v is not None:
            v.Start = float(val)

    add_tr_constraints(m, var_map, xstar_sel, delta)

    m.optimize()

    status = getattr(m, "Status", None)
    runtime = getattr(m, "Runtime", None)
    obj = None
    gap = None
    try:
        if hasattr(m, "SolCount") and m.SolCount and m.SolCount > 0:
            obj = float(m.ObjVal)
    except Exception:
        obj = None
    try:
        gap = float(m.MIPGap)
    except Exception:
        gap = None

    return {
        "file": str(mps_path),
        "status": int(status) if status is not None else None,
        "runtime": float(runtime) if runtime is not None else None,
        "obj": obj,
        "gap": gap,
        "alpha_groups": int(n_groups),
        "selected_groups": int(n_selected),
        "delta": int(delta),
        "error": None,
    }


def worker_main(wid, in_q, out_q, args):
    seed = int(args.seed) + int(wid)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    policy, device = build_policy()
    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    policy.load_state_dict(state)
    policy.eval()

    while True:
        mps_path = in_q.get()
        if mps_path is None:
            break
        try:
            res = solve_one(
                mps_path=mps_path,
                policy=policy,
                device=device,
                p_top=args.p_top,
                time_limit=args.time_limit,
                threads=args.threads,
                logdir=args.logdir,
            )
        except Exception as e:
            res = {
                "file": str(mps_path),
                "status": None,
                "runtime": None,
                "obj": None,
                "gap": None,
                "alpha_groups": None,
                "selected_groups": None,
                "delta": None,
                "error": f"{type(e).__name__}: {e}",
            }
        out_q.put(res)


def main():
    ap = argparse.ArgumentParser(description="Batch Predict-and-Search with multiprocessing over a directory of MPS files.")
    ap.add_argument("--mps-dir", required=True, help="Directory containing .mps files")
    ap.add_argument("--pattern", default="*.mps", help="Glob pattern within --mps-dir (default: *.mps)")
    ap.add_argument("--model", required=True, help="Path to trained GNN .pth")
    ap.add_argument("--p-top", type=float, default=20.0, help="Pick top p%% groups by (p1-p2) confidence")
    ap.add_argument("--time-limit", type=float, default=3600, help="Gurobi time limit (s)")
    ap.add_argument("--threads", type=int, default=1, help="Gurobi Threads per process")
    ap.add_argument("--seed", type=int, default=0, help="Random seed base")
    ap.add_argument("--run-id", default="default", help="Run id for outputs (folder name under ./experiments/)")
    ap.add_argument("--logdir", default=None, help="Folder to write solver logs (default: experiments/<run-id>/search/logs)")
    default_workers = max(1, min(8, os.cpu_count() or 1))
    ap.add_argument("--processes", type=int, default=default_workers, help="Number of worker processes")
    args = ap.parse_args()

    if args.logdir is None:
        args.logdir = str(Path("experiments") / str(args.run_id) / "search" / "logs")

    mps_dir = Path(args.mps_dir)
    if not mps_dir.is_dir():
        raise SystemExit(f"--mps-dir not found: {mps_dir}")
    files = sorted(glob.glob(str(mps_dir / args.pattern)))
    if not files:
        raise SystemExit(f"No files matched in {mps_dir} with pattern {args.pattern}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    in_q: mp.Queue = mp.Queue()
    out_q: mp.Queue = mp.Queue()

    for f in files:
        in_q.put(f)
    for _ in range(args.processes):
        in_q.put(None)

    procs = []
    for wid in range(args.processes):
        p = mp.Process(target=worker_main, args=(wid, in_q, out_q, args), daemon=True)
        p.start()
        procs.append(p)

    results = []
    for _ in range(len(files)):
        res = out_q.get()
        results.append(res)

    for p in procs:
        p.join()

    ok = sum(1 for r in results if not r.get("error"))
    print(f"Processed {len(results)} files. Success: {ok}, Errors: {len(results)-ok}. Logs: {args.logdir}")


if __name__ == "__main__":
    main()
