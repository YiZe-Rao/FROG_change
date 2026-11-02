# ============================================================
# 二跳关系与结构约束模块
# 作用：定义“共现型”与“补全型”二跳卷积，以及关系判别损失。
# ============================================================
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch import Tensor
from torch_geometric.typing import Adj, OptPairTensor, PairTensor, Size

class CoOccurrenceConv(MessagePassing):
    """将两个实体共享一个中间节点的二跳关系建模为“共现型卷积”。"""
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "sum",
        normalize: bool = False,
        root_weight: bool = False,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        # Optional projection (same as SAGEConv)
        if self.project:
            if in_channels[0] <= 0:
                raise ValueError("`project=True` does not support lazy init.")
            self.lin = nn.Linear(in_channels[0], in_channels[0], bias=True)
        # message = (x_i, h_a, x_j) → 3F
        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0] * 3
            )
        else:
            aggr_out_channels = in_channels[0] * 3
        self.lin_l = nn.Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        x_mid: Tensor,
        edge_index_mix: Tuple[Adj, Tensor],
        size: Size = None,
    ) -> Tensor:
        edge_index, central_node_index = edge_index_mix
        if isinstance(x, Tensor):
            x = (x, x)
        if self.project:
            x = (self.lin(x[0]).relu(), x[1])
        edge_attr = x_mid[central_node_index]  # h_a
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            size=size,
        )
        out = self.lin_l(out)
        if self.root_weight and x[1] is not None:
            out = out + self.lin_r(x[1])
        if self.normalize:
            out = nn.functional.normalize(out, p=2., dim=-1)
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        if edge_attr is None:
            edge_attr = torch.zeros_like(x_j)
        return torch.cat([x_i, edge_attr, x_j], dim=-1)


class CompletionWithCenterConv(MessagePassing):
    """将二跳关系视为“以中间节点为中心的补全过程”，利用中心节点信息进行编码。"""
    def __init__(self, in_channels, out_channels, aggr):
        super().__init__(aggr=aggr)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, 1),  # f(hb||hc) -> scalar
            nn.Sigmoid()
        )
        self.msg_mlp_inner = nn.Linear(in_channels * 2, out_channels)
        self.msg_mlp_outer = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, x_mid, edge_index_mix):
        edge_index, central_node_index = edge_index_mix
        return self.propagate(edge_index, x=x, x_mid=x_mid, central_node_index=central_node_index)

    def message(self, x_i, x_j, x_mid, central_node_index=None):
        if central_node_index is not None:
            h_b = x_mid[central_node_index]
            h_bc = torch.cat([h_b, x_j], dim=-1)
        else:
            raise ValueError("central_node_index must be provided for Completion convolution.")

        gate = self.gate_mlp(h_bc)                 # sigmoid(f(hb||hc))
        inner_msg = self.msg_mlp_inner(h_bc)       # W(hb||hc)
        gated_msg = gate * inner_msg               # sigmoid * W(hb||hc)
        ha_msg = torch.cat([x_i, gated_msg], dim=-1)
        return self.msg_mlp_outer(ha_msg)          # W(ha || gated_msg)
    

class RelJudgeDiff(nn.Module):
    """关系判别器：判断两个节点间的关系差异是否更像正向关系而不是随机负例。"""
    def __init__(self, emb_dim, hidden_dim=128, tau=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.tau = tau

    def score(self, emb_i, emb_j):
        diff = emb_j - emb_i
        return self.scorer(diff).squeeze(-1)

    def loss(self, emb1, emb2):
        emb_i = emb1
        emb_pos = emb2
        perm = torch.randperm(emb2.size(0))
        emb_neg = emb2[perm]
        pos_score = self.score(emb_i, emb_pos) / self.tau   # [B]
        neg_score = self.score(emb_i, emb_neg) / self.tau   # [B]

        logits = torch.stack([pos_score, neg_score], dim=1) # [B, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

    def loss_multi(self, emb1, emb2, K=5):
        B, D = emb1.size()
        device = emb1.device
        # positive
        pos_score = self.score(emb1, emb2) / self.tau   # [B]
        # sample K negatives per anchor
        neg_indices = []
        for i in range(B):
            candidates = torch.cat([
                torch.arange(i, device=device),
                torch.arange(i + 1, B, device=device)
            ])
            neg_indices.append(
                candidates[torch.randperm(B - 1)[:K]]
            )
        neg_indices = torch.stack(neg_indices)          # [B, K]
        emb_neg = emb2[neg_indices]                      # [B, K, D]

        # score negatives
        emb1_exp = emb1.unsqueeze(1).expand(B, K, D)
        neg_score = self.score(
            emb1_exp.reshape(-1, D),
            emb_neg.reshape(-1, D)
        ).view(B, K) / self.tau

        # logits
        logits = torch.cat(
            [pos_score.unsqueeze(1), neg_score],
            dim=1
        )                                                # [B, 1+K]

        labels = torch.zeros(B, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

class FDSubspace(nn.Module):
    """特征子空间约束，用于让关系差分在某子空间中保持稳定。"""
    def __init__(self, emb_dim, sub_dim):
        super().__init__()
        self.P = nn.Parameter(torch.randn(emb_dim, sub_dim))
        self.relation_bias = nn.Parameter(torch.randn(emb_dim))

    def project(self, delta):
        P = self.P       # [d, k]
        proj = delta @ P @ P.t()  # [B, d]
        return proj

    def loss(self, h_x, h_y):
        delta = h_y - h_x - self.relation_bias
        proj = self.project(delta)
        err = ((delta - proj) ** 2).sum(dim=-1)
        return err.mean()


