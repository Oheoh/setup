import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path


plot_configuration = {
    "time_horizon": 3600,
    "plot_seconds": 200,# 选择x轴的范围
    "y_lim": (0, 0.5),  # 选择y轴的范围
    "colormap": "turbo",
    "files": {
        "Gurobi": {"path": "experiments/myrun/grb_solve/logs"},
        "Ours": {"path": "experiments/withsec_CE_1e-4_withfeture/search/logs/ptop-75", "color": "red"},
    },
    "save_path": "./duibi.png",
}

_PTOP_SUFFIX_RE = re.compile(r"_ptop-[0-9.]+$")


def _instance_key_from_log_name(filename: str) -> str:
    """
    从 log 文件名推断实例 key。
    - 去掉尾部 `_ptop-xx(.x)`（如果有）
    """
    stem = Path(filename).stem
    stem = _PTOP_SUFFIX_RE.sub("", stem)
    return stem


def _parse_gurobi_log_to_curve(log_path: str, time_horizon: int) -> np.ndarray:
    """
    解析单个 Gurobi .log，提取 incumbent objective 随时间变化，并返回长度为 time_horizon 的曲线。
    解析规则：
    - 只解析 MIP progress table 中含 '%' 且以 '<t>s' 结尾的行
    - incumbent 取倒数第5列（通常是 Incumbent）
    - 累计取 best-so-far（默认按最小化）
    - 对缺失时间点做 forward fill，开头做 backfill（如果从未出现 incumbent 则全 NaN）
    """
    updates: dict[int, float] = {}
    best_so_far: float | None = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    start = 0
    for i, line in enumerate(lines):
        if "Gurobi Optimizer version" in line:
            start = i

    for line in lines[start:]:
            s = line.strip()
            if not s or "%" not in s or not s.endswith("s"):
                continue

            toks = s.split()
            if len(toks) < 5:
                continue

            t_tok = toks[-1]
            if not t_tok.endswith("s"):
                continue
            t_str = t_tok[:-1]
            if not t_str.isdigit():
                continue
            t = int(t_str)

            inc_tok = toks[-5]
            try:
                inc = float(inc_tok)
            except Exception:
                continue

            if best_so_far is None:
                best_so_far = inc
            else:
                best_so_far = min(best_so_far, inc)

            updates[t] = best_so_far

    curve = np.full((time_horizon,), np.nan, dtype=np.float64)
    for t, val in updates.items():
        if 1 <= t <= time_horizon:
            curve[t - 1] = val

    # forward fill
    for i in range(1, time_horizon):
        if np.isnan(curve[i]):
            curve[i] = curve[i - 1]

    # backfill prefix
    if np.isnan(curve[0]):
        idx = np.where(~np.isnan(curve))[0]
        if idx.size > 0:
            curve[: idx[0]] = curve[idx[0]]

    return curve


def _load_primals_from_log_dir(
    log_dir: str,
    time_horizon: int,
) -> tuple[list[str], np.ndarray]:
    """
    从目录读取 *.log，输出:
    - keys: 实例 key 列表
    - primals: shape=(n_instances, time_horizon)
    """
    p = Path(log_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"log dir not found: {log_dir}")

    logs = sorted(x for x in p.glob("*.log") if x.is_file())
    if not logs:
        raise FileNotFoundError(f"no .log files found in: {log_dir}")

    # 若同一实例出现多个 log，保留 mtime 最新的那一个。
    chosen: dict[str, tuple[float, Path]] = {}
    for lp in logs:
        key = _instance_key_from_log_name(lp.name)
        mtime = float(lp.stat().st_mtime)
        prev = chosen.get(key)
        if prev is None or mtime > prev[0]:
            chosen[key] = (mtime, lp)

    items: list[tuple[str, np.ndarray]] = []
    for key, (_, lp) in chosen.items():
        curve = _parse_gurobi_log_to_curve(str(lp), time_horizon=time_horizon)
        items.append((key, curve))

    items.sort(key=lambda x: x[0])
    keys = [k for k, _ in items]
    primals = np.stack([c for _, c in items], axis=0)
    return keys, primals

def draw_primal_curves_custom(plot_config: dict):
    """
    绘制原始间隙曲线，支持通过一个配置字典灵活定义所有绘图参数。

	    Args:
	        plot_config (dict): 包含所有绘图设置的字典。
	                            示例结构：
	                            {
	                                'files': {
	                                    'Gurobi': {'path': 'path/to/gurobi_dir/', 'color': 'black', 'marker': 'o', 'linestyle': '-'},
	                                    'Method A': {'path': 'path/to/method_a_logs_dir/', 'color': '#d95319', 'marker': 'd', 'linestyle': '--'},
	                                    ...
	                                },
	                                'save_path': 'path/to/output/primal_gap.png' # 可选：指定图片保存的完整路径或目录
	                            }
    """
    
    time_horizon = int(plot_config.get("time_horizon", 1000))
    plot_seconds = int(plot_config.get("plot_seconds", min(1000, time_horizon)))
    file_configs = plot_config.get('files', {})
    output_save_path = plot_config.get('save_path') # 获取保存路径
    y_lim = plot_config.get("y_lim", (0, 0.002))  # 兼容旧逻辑；如果设为 None 则不强制设置

    if not file_configs:
        print("错误：未在配置中提供任何文件信息。")
        return

    # 动态加载各方法的原始解数据
    raw_data_by_label: dict[str, tuple[list[str], np.ndarray]] = {}
    for label, config in file_configs.items():
        raw_path = config.get('path')
        if not raw_path:
            print(f"警告：标签 '{label}' 未指定文件路径，将跳过。")
            continue

        raw_path = str(raw_path)
        try:
            keys, primals = _load_primals_from_log_dir(
                raw_path,
                time_horizon=time_horizon,
            )
            raw_data_by_label[label] = (keys, primals)
        except Exception as e:
            print(f"警告：目录 '{raw_path}' 无法作为 log 目录解析，错误信息：{e}，将跳过标签 '{label}'。")
            continue

    if not raw_data_by_label:
        print("错误：所有指定文件均不存在或无法加载。请检查文件路径和文件内容。")
        return

    # 对齐实例：取所有方法共有的实例 key（避免不同目录缺 log 导致行数不一致）
    label_list = list(raw_data_by_label.keys())
    common_keys = set(raw_data_by_label[label_list[0]][0])
    for lab in label_list[1:]:
        common_keys &= set(raw_data_by_label[lab][0])
    common_keys = sorted(common_keys)

    if not common_keys:
        print("错误：不同方法之间没有可对齐的共同实例（common_keys 为空）。")
        print("提示：请检查各目录下 .log 文件名是否能映射到同一个实例 key。")
        for lab in label_list:
            keys_preview = raw_data_by_label[lab][0][:5]
            print(f"      - {lab} keys 示例: {keys_preview}")
        return

    data_dict: dict[str, np.ndarray] = {}
    for lab in label_list:
        keys, primals = raw_data_by_label[lab]
        if len(keys) != primals.shape[0]:
            print(f"警告：标签 '{lab}' keys 数量与 primals 行数不一致，将跳过。")
            continue
        key_to_row = {k: i for i, k in enumerate(keys)}
        rows = [key_to_row[k] for k in common_keys if k in key_to_row]
        if len(rows) != len(common_keys):
            print(f"警告：标签 '{lab}' 缺少部分实例，将按 common_keys 取交集行。")
        data_dict[lab] = primals[rows, :]

    if not data_dict:
        print("错误：对齐后没有可用数据。")
        return


    # 提取各方法最终解作为BKV的候选
    all_last_vals = [data[:, -1] for data in data_dict.values()]

    if not all_last_vals:
        print("错误：没有有效数据可用于BKV计算。")
        return

    BKV_candidates = np.stack(all_last_vals, axis=1)

    # 默认按最小化取 BKV
    BKV = np.nanmin(BKV_candidates, axis=1, keepdims=True)

    # 如果某些实例在所有方法下都没有 incumbent，会导致该行全 NaN；这些实例应从统计中剔除
    valid_instance_mask = ~np.isnan(BKV[:, 0])
    num_invalid = int((~valid_instance_mask).sum())
    if num_invalid > 0:
        print(f"提示：有 {num_invalid} 个实例在所有方法下都没有可用 incumbent（BKV 为 NaN），已从统计中忽略。")
    if not np.any(valid_instance_mask):
        print("错误：所有实例都没有可用 incumbent（BKV 全为 NaN），无法计算 primal gap。")
        return

    # 计算平均原始间隙 (PAG)
    def calc_pag(data):
        # 避免除以零，并处理BKV为负数的情况
        d = data[valid_instance_mask, :]
        bkv = BKV[valid_instance_mask, :]
        return np.nanmean(np.abs(d - bkv) / np.abs(bkv + 1e-10), axis=0)

    pag = {k: calc_pag(v) for k, v in data_dict.items()}

    # --- 修复：将 BKV_OBJ 的计算提前到绘图之前 ---
    BKV_OBJ = float(np.nanmean(BKV[valid_instance_mask, 0], axis=0))
    # --- 修复结束 ---

    # 设置绘图参数
    plt.figure(figsize=(12, 6))
    
    # 默认标记和线条样式（颜色用 colormap 自动生成，避免曲线多时“看起来重复”）
    default_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', 'p', '<', '>', 'd']
    default_linestyles = ['-', '--', '-.', ':']

    time_axis = np.arange(1, plot_seconds + 1)
    
    # 决定最终要画的曲线
    plot_labels: list[str] = [lab for lab in file_configs.keys() if lab in pag]

    # 为要绘制的曲线生成足够多的区分色
    cmap_name = str(plot_config.get("colormap", "turbo"))
    try:
        cmap = plt.get_cmap(cmap_name)
    except Exception:
        print(f"警告：colormap='{cmap_name}' 无法使用，将回退到 'turbo'。")
        cmap = plt.get_cmap("turbo")
    n_series = max(1, len(plot_labels))
    auto_colors = [cmap(i / (n_series - 1 if n_series > 1 else 1)) for i in range(n_series)]

    # 动态绘制曲线
    for i, label in enumerate(plot_labels):
        config = file_configs.get(label, {})
        
        # 从配置中获取样式，如果未指定则使用默认循环样式
        current_color = config.get('color', auto_colors[i])
        # 让 marker/linestyle 的组合尽量不重复（而不是各自简单取 mod）
        current_marker = config.get('marker', default_markers[i % len(default_markers)])
        current_linestyle = config.get(
            'linestyle',
            default_linestyles[(i // len(default_markers)) % len(default_linestyles)],
        )

        plt.plot(time_axis, pag[label][:plot_seconds], 
                 linestyle=current_linestyle, 
                 marker=current_marker,
                 markevery=100,
                 markersize=8,
                 linewidth=2,
                 color=current_color,
                 label=label)

    # 设置坐标轴
    if y_lim is not None:
        plt.ylim(*y_lim)  # 根据需要调整Y轴范围
    plt.xlim(1, plot_seconds)
    if plot_seconds >= 1000:
        plt.xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.grid(True)
    
    # 动态设置图例
    plt.legend(prop={'size': 11}, loc='best', handlelength=2.5)
    
    plt.xlabel('Solving time (s)', fontsize=12)
    plt.ylabel('Primal gap', fontsize=12)
    plt.tight_layout()
    
    if output_save_path:
        output_save_path = str(output_save_path)
        if not os.path.splitext(output_save_path)[1]:
            print(f"错误：save_path 必须是完整的文件路径（例如 ./duibi.png），当前：{output_save_path}")
            return
        output_dir = os.path.dirname(output_save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_save_path, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存至: {output_save_path}")
    else:
        print("\n未指定保存路径，图片将不会保存到文件。")
    
    plt.show()

    # --- 打印最终的原始间隙信息（对齐优化） ---
    # 计算最大标签长度以进行对齐
    all_display_labels = list(data_dict.keys())
    max_label_len = max(len(label) for label in all_display_labels) if all_display_labels else 0
    max_label_len = max(max_label_len, len("BKV")) # 确保BKV标签也考虑在内

    # 定义第一列的宽度（标签 + 额外填充）
    first_col_width = max_label_len + 2 # 标签后加2个空格

    # 定义数值部分的显示宽度，确保对齐
    obj_val_display_width = 6 # 例如 "12.03"
    gap_val_display_width = 6 # 例如 "1.05"
    gain_val_display_width = 7 # 例如 "16.62%"

    # 定义固定前缀
    obj_prefix = "OBJ:"
    obj_gap_abs_prefix = "OBJ_gap_abs:"
    gain_prefix = ", gain:"

    print(f"\nPrimal gap at {plot_seconds}s:")
    if "baseline_label" in plot_config:
        print(f"Baseline: {plot_config.get('baseline_label')}")

    # 打印 BKV 行
    bkv_label_formatted = f"{'BKV:':<{first_col_width}}"
    bkv_obj_formatted = f"{obj_prefix} {BKV_OBJ:>{obj_val_display_width}.2f}"
    print(f"{bkv_label_formatted}{bkv_obj_formatted}")

    # 获取基线数据（第一个文件）
    labels = list(data_dict.keys())
    if not labels:
        print("没有有效数据可用于最终结果打印。")
        return

    baseline_label = str(plot_config.get("baseline_label", labels[0]))
    if baseline_label not in data_dict:
        print(f"警告：baseline_label='{baseline_label}' 不在数据中，将使用默认基线 '{labels[0]}'。")
        baseline_label = labels[0]
    baseline_data = data_dict[baseline_label]
    report_idx = min(plot_seconds, time_horizon) - 1
    baseline_obj = float(np.nanmean(baseline_data[valid_instance_mask, report_idx], axis=0))
    baseline_gap_abs = baseline_obj - BKV_OBJ

    # 打印基线行
    baseline_label_formatted = f"{baseline_label + ':':<{first_col_width}}"
    baseline_obj_formatted = f"{obj_prefix} {baseline_obj:>{obj_val_display_width}.2f}"
    baseline_gap_abs_formatted = f"{obj_gap_abs_prefix} {baseline_gap_abs:>{gap_val_display_width}.2f}"
    print(f"{baseline_label_formatted}{baseline_obj_formatted} {baseline_gap_abs_formatted}") # OBJ和OBJ_gap_abs之间一个空格

    # 打印其他行
    for label in labels:
        if label == baseline_label:
            continue
        data = data_dict[label]
        obj = float(np.nanmean(data[valid_instance_mask, report_idx], axis=0))
        gap_abs = obj - BKV_OBJ
        
        # gain 定义：相对基线 gap_abs 的改善百分比（gap_abs 越小越好）
        # 若基线 gap_abs 过小（≈0），百分比会失去意义；此时：
        # - 若当前 gap_abs 也≈0：gain=0
        # - 否则：gain=NaN（表示不可比）
        gain_eps = float(plot_config.get("gain_eps", 1e-6))
        gain_percent = float("nan")
        if np.isnan(baseline_gap_abs) or np.isnan(gap_abs):
            gain_percent = float("nan")
        elif abs(baseline_gap_abs) <= gain_eps:
            gain_percent = 0.0 if abs(gap_abs) <= gain_eps else float("nan")
        else:
            gain_percent = -(gap_abs - baseline_gap_abs) / baseline_gap_abs * 100

        current_label_formatted = f"{label + ':':<{first_col_width}}"
        current_obj_formatted = f"{obj_prefix} {obj:>{obj_val_display_width}.2f}"
        current_gap_abs_formatted = f"{obj_gap_abs_prefix} {gap_abs:>{gap_val_display_width}.2f}"
        current_gain_formatted = f"{gain_prefix} {gain_percent:>{gain_val_display_width}.2f}%"

        print(f"{current_label_formatted}{current_obj_formatted} {current_gap_abs_formatted} {current_gain_formatted}") # 各部分之间一个空格



if __name__ == "__main__":
    draw_primal_curves_custom(plot_configuration)
