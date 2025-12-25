"""
Two-phase batch Gurobi run (primary objective + alpha secondary), multiprocess, unified outputs,
with separate time limits for primary and secondary objectives.
"""

import argparse
import os
import re
from pathlib import Path
import multiprocessing as mp
import pickle

import numpy as np
import gurobipy as gp


def build_secondary_alpha_objective(m):
    alpha_pat = re.compile(r"^alpha_(\d+)_(\d+)_(\d+)_(\d+)$")
    mvars = m.getVars()

    alpha_vars = []  # (var, l, t, v)
    t_values = []
    v_domain_by_l = {}
    for v in mvars:
        mobj = alpha_pat.fullmatch(v.VarName)
        if mobj is None:
            continue
        l_idx = int(mobj.group(1))
        t_idx = int(mobj.group(3))
        v_idx = int(mobj.group(4))
        alpha_vars.append((v, l_idx, t_idx, v_idx))
        t_values.append(t_idx)
        v_domain_by_l.setdefault(l_idx, []).append(v_idx)

    if not alpha_vars:
        return gp.LinExpr(), {}

    uniq_t = sorted(set(t_values))
    T_count = len(uniq_t)
    t_weights = {t: (T_count - idx) for idx, t in enumerate(uniq_t)}

    expr = gp.LinExpr()
    c_cache = {}
    for l_idx, v_list in v_domain_by_l.items():
        uniq_v = sorted(set(v_list))
        Vcnt = len(uniq_v)
        if Vcnt <= 1:
            for vv in uniq_v:
                c_cache[(l_idx, vv)] = 0.0
        else:
            ord_map = {vv: (i + 1) for i, vv in enumerate(uniq_v)}
            denom = float(Vcnt - 1)
            for vv in uniq_v:
                c_cache[(l_idx, vv)] = (ord_map[vv] - 1) / denom

    for var, l_idx, t_idx, v_idx in alpha_vars:
        rho = float(t_weights[t_idx])
        c_lv = float(c_cache.get((l_idx, v_idx), 0.0))
        coeff = rho * c_lv
        if coeff != 0.0:
            expr.addTerms(coeff, var)

    return expr, t_weights


def run_two_phase(
    file_path,
    outroot_path,
    base,
    primary_time_limit=3000,
    secondary_time_limit=600,
    threads=1,
    pool_solutions=0,
    solution_limit=0,
    primary_epsilon=0.0,
):
    gp.setParam("LogToConsole", 0)

    log_dir = os.path.join(outroot_path, "log")
    sol_dir = os.path.join(outroot_path, "sol")
    bg_dir = os.path.join(outroot_path, "bg")
    for d in (log_dir, sol_dir, bg_dir):
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(log_dir, f"{base}.log")
    sol_path = os.path.join(sol_dir, f"{base}.sol")
    bg_path = os.path.join(bg_dir, f"{base}.bg")

    m = gp.read(file_path)
    m.Params.LogFile = log_path

    m.Params.Heuristics = 1
    m.Params.Threads = threads
    m.Params.MIPFocus = 1

    primary = m.getObjective()
    m.setObjective(primary, gp.GRB.MINIMIZE)
    m.Params.TimeLimit = primary_time_limit
    m.optimize()

    zstar = float(m.ObjVal)

    sec_expr, _ = build_secondary_alpha_objective(m)
    if primary_epsilon < 0:
        primary_epsilon = 0.0
    m.addConstr(primary <= zstar + primary_epsilon, name="fix_primary_le")
    m.addConstr(primary >= zstar - primary_epsilon, name="fix_primary_ge")

    m.setObjective(sec_expr, gp.GRB.MINIMIZE)
    if pool_solutions and pool_solutions > 1:
        m.Params.PoolSolutions = int(pool_solutions)
        m.Params.PoolSearchMode = 2
        m.Params.PoolGap = 0
    if solution_limit and solution_limit > 0:
        m.Params.SolutionLimit = int(solution_limit)

    m.Params.TimeLimit = secondary_time_limit
    m.optimize()
    _populate = getattr(m, "populate", None)
    if callable(_populate) and pool_solutions and pool_solutions > 1:
        _populate()

    from helper import get_a_new2

    A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(file_path)
    with open(bg_path, "wb") as bf:
        pickle.dump([A, v_map, v_nodes, c_nodes, b_vars], bf)

    solc = m.SolCount
    mvars = m.getVars()
    var_names = [v.VarName for v in mvars]

    if solc == 0:
        with open(sol_path, "wb") as f:
            pickle.dump(
                {
                    "var_names": [],
                    "sols": [],
                    "objs": [],
                    "sec_objs": [],
                    "primary_target": zstar,
                },
                f,
            )
        return

    sols = []
    primary_vals = []
    secondary_vals = []

    for i in range(solc):
        m.Params.SolutionNumber = i
        try:
            x = np.array(m.Xn, dtype=np.float32)
        except Exception:
            x = np.array([v.X for v in mvars], dtype=np.float32)
        sols.append(x)
        primary_vals.append(float(primary.getValue()))
        secondary_vals.append(float(sec_expr.getValue()))

    sols_arr = np.stack(sols) if sols else np.empty((0, len(mvars)), dtype=np.float32)
    primary_arr = np.array(primary_vals, dtype=np.float32)
    secondary_arr = np.array(secondary_vals, dtype=np.float32)

    sol_data = {
        "var_names": var_names,
        "sols": sols_arr,
        "objs": primary_arr,
        "sec_objs": secondary_arr,
        "primary_target": zstar,
    }
    with open(sol_path, "wb") as f:
        pickle.dump(sol_data, f)


def run_one_phase(
    file_path,
    outroot_path,
    base,
    time_limit=3600,
    threads=1,
    pool_solutions=0,
    solution_limit=0,
):
    gp.setParam("LogToConsole", 0)

    log_dir = os.path.join(outroot_path, "log")
    sol_dir = os.path.join(outroot_path, "sol")
    bg_dir = os.path.join(outroot_path, "bg")
    for d in (log_dir, sol_dir, bg_dir):
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(log_dir, f"{base}.log")
    sol_path = os.path.join(sol_dir, f"{base}.sol")
    bg_path = os.path.join(bg_dir, f"{base}.bg")

    m = gp.read(file_path)
    m.Params.LogFile = log_path
    m.Params.Heuristics = 1
    m.Params.Threads = threads
    m.Params.MIPFocus = 1
    m.Params.TimeLimit = time_limit
    if pool_solutions and pool_solutions > 1:
        m.Params.PoolSolutions = int(pool_solutions)
        m.Params.PoolSearchMode = 2
        m.Params.PoolGap = 0
    if solution_limit and solution_limit > 0:
        m.Params.SolutionLimit = int(solution_limit)

    primary = m.getObjective()
    m.setObjective(primary, gp.GRB.MINIMIZE)
    m.optimize()
    _populate = getattr(m, "populate", None)
    if callable(_populate) and pool_solutions and pool_solutions > 1:
        _populate()

    zstar = float(m.ObjVal) if getattr(m, "SolCount", 0) else float("nan")

    from helper import get_a_new2

    A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(file_path)
    with open(bg_path, "wb") as bf:
        pickle.dump([A, v_map, v_nodes, c_nodes, b_vars], bf)

    solc = m.SolCount
    mvars = m.getVars()
    var_names = [v.VarName for v in mvars]

    if solc == 0:
        with open(sol_path, "wb") as f:
            pickle.dump(
                {
                    "var_names": [],
                    "sols": [],
                    "objs": [],
                    "sec_objs": [],
                    "primary_target": zstar,
                },
                f,
            )
        return

    sols = []
    primary_vals = []
    secondary_vals = []
    for i in range(solc):
        m.Params.SolutionNumber = i
        try:
            x = np.array(m.Xn, dtype=np.float32)
        except Exception:
            x = np.array([v.X for v in mvars], dtype=np.float32)
        sols.append(x)
        primary_vals.append(float(primary.getValue()))
        secondary_vals.append(0.0)

    sols_arr = np.stack(sols) if sols else np.empty((0, len(mvars)), dtype=np.float32)
    primary_arr = np.array(primary_vals, dtype=np.float32)
    secondary_arr = np.array(secondary_vals, dtype=np.float32)

    sol_data = {
        "var_names": var_names,
        "sols": sols_arr,
        "objs": primary_arr,
        "sec_objs": secondary_arr,
        "primary_target": zstar,
    }
    with open(sol_path, "wb") as f:
        pickle.dump(sol_data, f)


def _worker(
    queue,
    data_root,
    outroot_path,
    mode,
    time_limit,
    primary_time_limit,
    secondary_time_limit,
    threads,
    pool_solutions,
    solution_limit,
    primary_epsilon,
):
    repo_root = Path(__file__).resolve().parents[0]
    if str(repo_root) not in os.sys.path:
        os.sys.path.insert(0, str(repo_root))

    for inst_path in iter(queue.get, None):
        rel_path = os.path.relpath(inst_path, data_root)

        p = Path(inst_path)
        parent_tag = p.parent.name or Path(data_root).name
        stem = p.stem
        base = f"{parent_tag}__{stem}" if parent_tag else stem

        mode_norm = str(mode).strip().lower()
        if mode_norm == "with_secondary_objective":
            run_two_phase(
                file_path=inst_path,
                outroot_path=outroot_path,
                base=base,
                primary_time_limit=primary_time_limit,
                secondary_time_limit=secondary_time_limit,
                threads=threads,
                pool_solutions=pool_solutions,
                solution_limit=solution_limit,
                primary_epsilon=primary_epsilon,
            )
        elif mode_norm == "without_secondary_objective":
            run_one_phase(
                file_path=inst_path,
                outroot_path=outroot_path,
                base=base,
                time_limit=time_limit,
                threads=threads,
                pool_solutions=pool_solutions,
                solution_limit=solution_limit,
            )
        else:
            raise ValueError(f"Unknown --mode: {mode}")
        print(f"[OK] {rel_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Batch solve with optional secondary alpha objective, using multiprocessing and unified outputs."
    )
    ap.add_argument("--data-root", required=True, help="Input root to search instances recursively")
    ap.add_argument("--out-root", required=True, help="Output root for logs/sols/bg (split by type; filenames prefixed with parent dir)")
    ap.add_argument("--mode", default="with_secondary_objective", choices=["with_secondary_objective", "without_secondary_objective"], help="Solve mode")
    ap.add_argument("--processes", type=int, default=32, help="Number of worker processes (default 32)")
    ap.add_argument("--threads", type=int, default=1, help="Threads per Gurobi solve (default 1)")
    ap.add_argument("--time-limit", type=int, default=3600, help="Time limit for primary-only solve (seconds)")
    ap.add_argument("--primary-time-limit", type=int, default=3000, help="Time limit for primary objective (seconds)")
    ap.add_argument("--secondary-time-limit", type=int, default=600, help="Time limit for secondary objective (seconds)")
    ap.add_argument("--pool-solutions", type=int, default=0, help="If >1, enable Gurobi solution pool with this capacity; default 0 = disabled")
    ap.add_argument("--solution-limit", type=int, default=0, help="Optional hard limit on number of solutions to search in Phase 2 (0=unset)")
    ap.add_argument("--primary-epsilon", type=float, default=0.0, help="Tolerance for fixing primary objective around z* in Phase 2")
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    files = sorted(str(p) for p in root.rglob("*") if p.is_file())
    print(f"Found {len(files)} instance(s) under {args.data_root}")
    if not files:
        return 0

    os.makedirs(args.out_root, exist_ok=True)

    queue = mp.Queue(maxsize=max(2 * args.processes, 64))
    workers = []
    for _ in range(max(args.processes, 1)):
        p = mp.Process(
            target=_worker,
            args=(
                queue,
                args.data_root,
                args.out_root,
                args.mode,
                args.time_limit,
                args.primary_time_limit,
                args.secondary_time_limit,
                args.threads,
                args.pool_solutions,
                args.solution_limit,
                args.primary_epsilon,
            ),
        )
        p.daemon = True
        p.start()
        workers.append(p)

    for path in files:
        queue.put(path)
    for _ in workers:
        queue.put(None)

    for p in workers:
        p.join()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
