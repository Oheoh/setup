import os
import re
import torch
import torch_geometric
import gzip
import pickle
import numpy as np
import time
import alpha_utils as au

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    Simple bipartite graph convolution between constraint nodes and variable nodes.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class BipartiteNodeData(torch_geometric.data.Data):
    """
    Node bipartite graph observation in a format understood by pyg DataLoader.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        Tell pyg how to increment indices when concatenating graphs.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, store, *args, **kwargs)


class GNNPolicy_position(torch.nn.Module):
    """
    Position-augmented GNN policy (without any self-attention).
    """

    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 26

        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
    ):
        reversed_edge_indices = torch.stack(
            [edge_indices[1], edge_indices[0]], dim=0
        )

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features,
            reversed_edge_indices,
            edge_features,
            constraint_features,
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features,
            reversed_edge_indices,
            edge_features,
            constraint_features,
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output


class GraphDataset_position(torch_geometric.data.Dataset):
    """
    Dataset for GNNPolicy_position (with positional encoding and c_lv / t features).
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData
        varNames = solData["var_names"]

        sols = solData["sols"][:50]
        objs = solData["objs"][:50]

        sols = np.round(sols, 0)
        return BG, sols, objs, varNames

    def get(self, index):
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars = BG

        constraint_features = c_nodes
        edge_indices = A._indices()
        variable_features = v_nodes

        try:
            C = constraint_features.size(0)
            V = variable_features.size(0)
            row = edge_indices[0]
            col = edge_indices[1]
            mask = (row >= 0) & (row < C) & (col >= 0) & (col < V)
            if mask.numel() > 0 and (~mask).any():
                edge_indices = torch.stack([row[mask], col[mask]], dim=0)
        except Exception:
            pass

        edge_values = A._values()
        if "mask" in locals() and mask.numel() == edge_values.numel():
            edge_values = edge_values[mask]
        edge_features = edge_values.unsqueeze(1)

        variable_features = postion_get(variable_features)
        c_lv = au.build_c_lv_feature(v_map).unsqueeze(1)
        t_feat = au.build_t_feature(v_map).unsqueeze(1)
        variable_features = torch.cat([variable_features, c_lv, t_feat], dim=1)
        variable_features = torch.nan_to_num(
            variable_features, nan=1.0, posinf=1.0, neginf=1.0
        )

        graph = BipartiteNodeData(
            constraint_features.float(),
            edge_indices.long(),
            edge_features.float(),
            variable_features.float(),
        )

        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames

        varname_map = au.build_varname_map(varNames, v_map)
        if varname_map.numel() > 0 and len(varNames) > 0:
            varname_map = varname_map.clamp_(0, len(varNames) - 1)
        alpha_idx, group_ptr = au.build_alpha_index_and_group_ptr(v_map)
        graph.varInds = [[varname_map], [alpha_idx], [group_ptr]]

        return graph


def postion_get(variable_features):
    dev = variable_features.device
    lens = variable_features.shape[0]
    feature_widh = 18  # support up to 2^18-1 positions

    positions = torch.arange(lens, device=dev, dtype=torch.long).unsqueeze(
        1
    )  # [N,1]
    shifts = torch.arange(feature_widh, device=dev, dtype=torch.long).unsqueeze(
        0
    )  # [1,W]
    position_feature = ((positions >> shifts) & 1).to(variable_features.dtype)  # [N,W]

    if not torch.is_floating_point(variable_features):
        variable_features = variable_features.float()
    return torch.cat([variable_features, position_feature], dim=1)
