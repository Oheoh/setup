import os
from typing import List, Tuple


def collect_pairs(root: str) -> List[Tuple[str, str]]:
    """Collect (.bg, .sol) pairs under root.

    Supports two layouts:
    1) Sibling dirs: <root>/bg/*.bg and <root>/sol/*.sol (same stem)
    2) Flat/recursive: .bg and .sol in same folder (same stem)
    """
    root = os.path.abspath(root)
    bg_dir = os.path.join(root, "bg")
    sol_dir = os.path.join(root, "sol")

    pairs: List[Tuple[str, str]] = []
    if os.path.isdir(bg_dir) and os.path.isdir(sol_dir):
        for fn in sorted(os.listdir(bg_dir)):
            if not fn.endswith(".bg"):
                continue
            stem = os.path.splitext(fn)[0]
            bg = os.path.join(bg_dir, fn)
            sol = os.path.join(sol_dir, f"{stem}.sol")
            if os.path.exists(sol):
                pairs.append((bg, sol))
            else:
                print(f"[WARN] missing .sol for {bg}")
        return pairs

    for dp, _ds, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".bg"):
                continue
            bg = os.path.join(dp, fn)
            stem = os.path.splitext(fn)[0]
            sol = os.path.join(dp, f"{stem}.sol")
            if os.path.exists(sol):
                pairs.append((bg, sol))
            else:
                print(f"[WARN] missing .sol for {bg}")
    pairs.sort()
    return pairs
