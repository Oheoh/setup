import re
from collections import OrderedDict
import torch

_PAT = re.compile(r"alpha_(\d+)_(\d+)_(\d+)_(\d+)")


def parse_alpha(name):
    match = _PAT.fullmatch(name)
    if not match:
        return None
    l, m, t, v = map(int, match.groups())
    return l, m, t, v


def build_varname_map(sol_var_names, v_map):
    name_to_idx = {n: i for i, n in enumerate(sol_var_names)}
    graph_names = list(v_map.keys()) if hasattr(v_map, "keys") else list(v_map)
    mapped = []
    for n in graph_names:
        idx = name_to_idx.get(n, None)
        if idx is None:
            idx = 0
        mapped.append(idx)
    return torch.tensor(mapped, dtype=torch.long)


def build_alpha_index_and_group_ptr(v_map):
    names_graph = list(v_map.keys()) if hasattr(v_map, "keys") else list(v_map)
    entries = []
    for idx, name in enumerate(names_graph):
        p = parse_alpha(name)
        if p is not None:
            l, m_, t, v = p
            entries.append((idx, (l, m_, t, v)))

    buckets = {}
    for idx, (l, m_, t, v) in entries:
        key = (l, m_, t)
        buckets.setdefault(key, []).append((idx, v))

    alpha_idx_list = []
    group_ptr = [0]
    for key in sorted(buckets.keys()):
        elems = sorted(buckets[key], key=lambda x: x[1])
        alpha_idx_list.extend([i for i, _ in elems])
        group_ptr.append(len(alpha_idx_list))

    alpha_idx = torch.tensor(alpha_idx_list, dtype=torch.long)
    group_ptr = torch.tensor(group_ptr, dtype=torch.long)
    return alpha_idx, group_ptr


def group_alphas_sorted(vnames, alpha_idx):
    buckets = {}
    for i in alpha_idx:
        ii = int(i)
        p = parse_alpha(vnames[ii])
        if p is None:
            continue
        l, m_, t, v = p
        buckets.setdefault((l, m_, t), []).append((ii, v))

    groups = OrderedDict()
    for key in sorted(buckets.keys()):
        groups[key] = [i for i, _ in sorted(buckets[key], key=lambda x: x[1])]
    return groups


def build_c_lv_feature(v_map):
    names_graph = list(v_map.keys()) if hasattr(v_map, "keys") else list(v_map)
    vset_by_l = {}
    idx_lv = [None] * len(names_graph)
    for i, name in enumerate(names_graph):
        p = parse_alpha(name)
        if p is None:
            continue
        l, _m, _t, v = p
        idx_lv[i] = (l, v)
        vset_by_l.setdefault(l, set()).add(v)

    ord_by_l = {}
    denom_by_l = {}
    for l, vset in vset_by_l.items():
        uniq_v = sorted(vset)
        if len(uniq_v) <= 1:
            denom_by_l[l] = 0.0
            ord_by_l[l] = {vv: 1 for vv in uniq_v}
        else:
            denom_by_l[l] = float(len(uniq_v) - 1)
            ord_by_l[l] = {vv: (i + 1) for i, vv in enumerate(uniq_v)}

    c = torch.zeros(len(names_graph), dtype=torch.float32)
    for i, lv in enumerate(idx_lv):
        if lv is None:
            continue
        l, v = lv
        denom = denom_by_l.get(l, 0.0)
        c[i] = 0.0 if denom <= 0.0 else (ord_by_l[l][v] - 1) / denom
    return c


def build_t_feature(v_map):
    names_graph = list(v_map.keys()) if hasattr(v_map, "keys") else list(v_map)
    t_feat = torch.zeros(len(names_graph), dtype=torch.float32)
    for i, name in enumerate(names_graph):
        p = parse_alpha(name)
        if p is not None:
            _l, _m, t, _v = p
            t_feat[i] = float(t)
    return t_feat
