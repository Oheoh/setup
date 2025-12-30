## 目录结构

```text
.
├── alpha_utils.py
├── pair_utils.py
├── GCN.py
├── helper.py
├── collect_labels.py
├── train.py
├── evaluate.py
├── PredictAndSearch.py
├── readme.md
├── data/
│   ├── instances/
│   │   └── <dataset_name>/
│   │       ├── train/                 # 原始训练实例（.mps）
│   │       └── test/                  # 原始测试实例（也可给 PredictAndSearch 用）
│   └── solvefiles/                    # collect_labels.py --out-root
│       └── <dataset_name>/
│           ├── train/                 # train.py --train-root
│           │   ├── bg/
│           │   ├── sol/
│           │   └── log/
│           └── test/                  # evaluate.py --test-root
│               ├── bg/
│               ├── sol/
│               └── log/
└── experiments/
    └── <run_id>/
        ├── model/
        │   ├── model_best.pth
        │   └── model_last.pth
        ├── training/
        │   ├── train.log
        │   └── loss_curve.png
        └── search/
            └── logs/
                └── <parent>_<stem>_ptop-XX.X.log
```
## Requirements

- OS: Linux
- python==3.11.13
- pytorch==2.5.1 (CUDA 12.1)
- torch-geometric==2.6.1
- pyscipopt==5.5.0
- gurobipy==12.0.3 (requires Gurobi + license)
- numpy==1.26.4
- matplotlib==3.8.0

## Installation

Create a new environment with Conda:

```bash
conda env create -f environment.yml
conda activate test
```


## 使用命令

### 1) 生成 solvefiles（.bg/.sol/.log）

带二级目标（一级目标默认求解 3000s，二级目标默认求解 600s）：

```bash
python collect_labels.py --data-root data/instances/<dataset_name>/train --out-root data/solvefiles/<dataset_name>/train --mode with_secondary_objective --processes 32
python collect_labels.py --data-root data/instances/<dataset_name>/test  --out-root data/solvefiles/<dataset_name>/test  --mode with_secondary_objective --processes 32
```

不带二级目标（仅一级目标，默认 3600s）：

```bash
python collect_labels.py --data-root data/instances/<dataset_name>/train --out-root data/solvefiles/<dataset_name>/train --mode without_secondary_objective --processes 32
python collect_labels.py --data-root data/instances/<dataset_name>/test  --out-root data/solvefiles/<dataset_name>/test  --mode without_secondary_objective --processes 32
```

### 2) 训练

```bash
python train.py --train-root data/solvefiles/<dataset_name>/train --run-id <run_id> --loss gumbel
```

### 3) 评估

```bash
python evaluate.py --test-root data/solvefiles/<dataset_name>/test --model experiments/<run_id>/model/model_best.pth
```

### 4) Predict-and-Search（下游任务）

```bash
python PredictAndSearch.py --mps-dir data/instances/<dataset_name>/test --model experiments/<run_id>/model/model_best.pth --run-id <run_id> --processes 32
```
