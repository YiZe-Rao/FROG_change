# ============================================================
# FROG 的图结构学习核心实现
# 作用：在异构图上进行消息传播，并融合一跳、二跳关系，以学习更优的图结构。
# ============================================================
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import coalesce
from utils import *
from hop_mode import CoOccurrenceConv, CompletionWithCenterConv, RelJudgeDiff, FDSubspace
import random

class HeteroGraphSAGE(torch.nn.Module):
    """异构图 SAGE：既做一跳聚合，也做二跳结构学习和门控融合。"""
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        two_hop_relations: set,
        channels: int,
        aggr: str = "sum",
        num_layers: int = 2,
        MAE: float = 0.99,
        dropout: float = 0.0,
        optim: str = 'both',
        fix: bool = False,
        fix_attn: float = 0.5
    ):
        super().__init__()
        self.num_layers = num_layers
        self.node_types = node_types
        self.edge_types = edge_types
        self.two_hop_relations, self.two_hop_neighbors = two_hop_relations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optim = 'both'

        # 1-hop 聚合层：对原始异构边做标准 SAGE 更新。
        self.convs = torch.nn.ModuleList()
        self.edge_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=None,
            )
            self.convs.append(conv)

            edge_conv = HeteroConv(  # node as edge: aggregate 2-hop neighbors
                {
                    edge_type: CoOccurrenceConv(channels, channels, aggr=aggr) 
                    if hop_type == 'co-occurrence' 
                    else CompletionWithCenterConv(channels, channels, aggr=aggr)
                    for edge_type, hop_type in self.two_hop_relations
                },
                aggr=None,
            )
            self.edge_convs.append(edge_conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        # 二跳特征融合门控：决定是保留一跳信息还是使用二跳增强信息。
        self.gates = torch.nn.ModuleDict()
        for rel, _ in self.two_hop_relations: 
            self.gates[tuple_2_str(rel)] = torch.nn.Sequential(
                torch.nn.Linear(channels * 2, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, 1),
                torch.nn.Sigmoid(),
            )

        self.path_attn_query = nn.Linear(channels, channels, bias=False)
        self.path_attn_key = nn.Linear(channels, channels, bias=False)
        self.path_attn_value = nn.Linear(channels, channels, bias=False)

        self.MAE = MAE

        self.fix = fix
        if self.fix: 
            if fix_attn >= -1e-9:
                self.attn = {tuple_2_str(key): fix_attn for key, _ in self.two_hop_relations}
            else:
                self.attn = {tuple_2_str(key): random.random() for key, _ in self.two_hop_relations}
                
        else:
            self.attn = {tuple_2_str(key): 0.5 for key, _ in self.two_hop_relations}

        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()
        self.path_attn_query.reset_parameters()
        self.path_attn_key.reset_parameters()
        self.path_attn_value.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        # x_dict[t]: [N_t, C] 节点特征；edge_index_dict[(src, rel, dst)]: [2, E] 局部边索引。
        # 先构造二跳邻接关系，以便在一跳和二跳信息之间做混合。
        edge_index_2hop_dict = self.get_2hop_neighbor_linear_optimized(edge_index_dict)

        for layer in range(self.num_layers):
            node_conv = self.convs[layer]
            edge_conv = self.edge_convs[layer]
            norm_dict = self.norms[layer]
            tmp_x_dict, tmp_edge_dict = {}, {}
            # 一跳邻居聚合：把直接相连节点的信息汇总到目标节点。
            # **重点改进位置**
            for rel, conv in node_conv.convs.items():
                src, edge, dst = rel
                edge_index = edge_index_dict[rel]  # 1-hop neighbors, e.g.: (results, f2p_racesID, races)
                out = conv((x_dict[src], x_dict[dst]), edge_index)
                # out: [N_dst, C]；N_dst 为当前批次该关系的目标端局部节点数。
                tmp_x_dict[rel] = out

            # 二跳邻居聚合：把中间实体视作“边”来做更高层的结构聚合。
            for rel, conv in edge_conv.convs.items():
                src, mid, dst = rel
                edge_index = edge_index_2hop_dict[rel]  # (二跳边索引 [2, E_2hop], 对应桥接节点 ID [E_2hop])
                if edge_index[1].numel() == 0:
                    continue
                out = conv((x_dict[src], x_dict[dst]), x_dict[mid], edge_index)
                # 二跳聚合输出与目标节点特征同形：[N_dst, C]。
                tmp_edge_dict[rel] = out
            
            # Fuse each one-hop/two-hop pair, then attend over valid paths per target node.
            x_out = {}
            path_candidates = {}
            path_masks = {}
            for key_node, x_node in tmp_x_dict.items():
                src1, _, dst1 = key_node
                has_valid_path = torch.zeros(
                    x_node.size(0), dtype=torch.bool, device=x_node.device
                )

                if dst1 not in x_out:
                    x_out[dst1] = torch.zeros_like(x_node)

                for key_edge, x_edge in tmp_edge_dict.items():
                    src2, mid2, dst2 = key_edge
                    if dst1 != dst2 or src1 != mid2:
                        continue

                    edge_index = edge_index_2hop_dict[key_edge][0]
                    route_mask = torch.zeros_like(has_valid_path)
                    route_mask[edge_index[1].unique()] = True
                    has_valid_path |= route_mask

                    gate_input = torch.cat([x_node, x_edge], dim=-1)
                    if self.training and not self.fix:
                        gate = self.gates[tuple_2_str(key_edge)](gate_input)
                        gate = (1 - self.MAE) * gate + self.MAE * self.attn[tuple_2_str(key_edge)]
                        self.attn[tuple_2_str(key_edge)] = gate.detach().clone().mean()
                    else:
                        gate = self.attn[tuple_2_str(key_edge)]

                    if self.optim == 'both':
                        candidate = gate * x_edge + (1 - gate) * x_node
                    else:
                        candidate = gate * x_edge + (1 - gate) * x_node.detach().clone()

                    path_candidates.setdefault(dst1, []).append(candidate)
                    path_masks.setdefault(dst1, []).append(route_mask)

                # Preserve the one-hop message where this relation has no valid 2-hop route.
                fallback_mask = (~has_valid_path).to(x_node.dtype).unsqueeze(-1)
                x_out[dst1] = x_out[dst1] + fallback_mask * x_node

            for dst, candidates in path_candidates.items():
                candidate_tensor = torch.stack(candidates, dim=0)
                valid_mask = torch.stack(path_masks[dst], dim=0)
                query = self.path_attn_query(x_dict[dst])
                keys = self.path_attn_key(candidate_tensor)
                scores = (keys * query.unsqueeze(0)).sum(dim=-1)
                scores = scores / (candidate_tensor.size(-1) ** 0.5)
                scores = scores.masked_fill(
                    ~valid_mask, torch.finfo(scores.dtype).min
                )

                weights = torch.softmax(scores, dim=0)
                weights = weights * valid_mask.to(weights.dtype)
                weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(
                    torch.finfo(weights.dtype).tiny
                )

                values = self.path_attn_value(candidate_tensor)
                attended = (weights.unsqueeze(-1) * values).sum(dim=0)
                x_out[dst] = x_out[dst] + attended

            for key in x_dict.keys():
                if key not in x_out.keys():
                    x_out[key] = torch.zeros_like(x_dict[key])
                x_out[key] = x_out[key] + x_dict[key]   # residual connection

            x_dict = {key: norm_dict[key](x) for key, x in x_out.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return x_dict


    # def get_hyperedge(self, edge_index_dict):
    #     """按共同中间节点生成异构超边的成员索引。

    #     每个共同中间节点对应一条超边；src、dst 类型的端点分别用 incidence
    #     索引表示，避免不同节点类型中相同的局部 ID 被误认为同一个节点。
    #     """
    #     hyperedge_dict = {}
    #     edge_search_dict = {
    #         (edge_type[0], edge_type[2]): edge_type for edge_type in self.edge_types
    #     }

    #     for (src, mid, dst), hop_type in self.two_hop_relations:
    #         if hop_type == "co-occurrence":
    #             rel_l = edge_search_dict.get((mid, src))
    #         else:  # completion
    #             rel_l = edge_search_dict.get((src, mid))
    #         rel_r = edge_search_dict.get((mid, dst))

    #         edge_l = edge_index_dict.get(rel_l) if rel_l is not None else None
    #         edge_r = edge_index_dict.get(rel_r) if rel_r is not None else None
    #         if edge_l is not None:
    #             device = edge_l.device
    #         elif edge_r is not None:
    #             device = edge_r.device
    #         else:
    #             device = torch.device("cpu")

    #         empty_incidence = torch.empty((2, 0), device=device, dtype=torch.long)
    #         empty_mid_index = torch.empty((0,), device=device, dtype=torch.long)
    #         if edge_l is None or edge_r is None:
    #             hyperedge_dict[(src, mid, dst)] = {
    #                 "src_index": empty_incidence,
    #                 "dst_index": empty_incidence.clone(),
    #                 "mid_index": empty_mid_index,
    #                 "num_hyperedges": 0,
    #             }
    #             continue

    #         if hop_type == "co-occurrence":
    #             mid_l, src_nodes = edge_l[0], edge_l[1]
    #         else:  # completion
    #             src_nodes, mid_l = edge_l[0], edge_l[1]
    #         mid_r, dst_nodes = edge_r[0], edge_r[1]

    #         # 只为左右两侧都至少有一个成员的中间节点建超边。
    #         common_mid = torch.unique(mid_l[torch.isin(mid_l, mid_r)], sorted=True)
    #         if common_mid.numel() == 0:
    #             hyperedge_dict[(src, mid, dst)] = {
    #                 "src_index": empty_incidence,
    #                 "dst_index": empty_incidence.clone(),
    #                 "mid_index": empty_mid_index,
    #                 "num_hyperedges": 0,
    #             }
    #             continue

    #         src_mask = torch.isin(mid_l, common_mid)
    #         dst_mask = torch.isin(mid_r, common_mid)
    #         src_index = torch.stack(
    #             [
    #                 src_nodes[src_mask],
    #                 torch.searchsorted(common_mid, mid_l[src_mask]),
    #             ],
    #             dim=0,
    #         )
    #         dst_index = torch.stack(
    #             [
    #                 dst_nodes[dst_mask],
    #                 torch.searchsorted(common_mid, mid_r[dst_mask]),
    #             ],
    #             dim=0,
    #         )

    #         # 每个 (节点, 超边) 关联只保留一次，防止输入重复边造成重复成员。
    #         src_index = torch.unique(src_index.t(), dim=0, sorted=True).t().contiguous()
    #         dst_index = torch.unique(dst_index.t(), dim=0, sorted=True).t().contiguous()
    #         hyperedge_dict[(src, mid, dst)] = {
    #             # incidence 格式 [2, M]：第 0 行是节点 ID，第 1 行是超边 ID。
    #             "src_index": src_index,
    #             "dst_index": dst_index,
    #             # hyperedge h 对应原图中间节点 mid_index[h]。
    #             "mid_index": common_mid,
    #             "num_hyperedges": common_mid.numel(),
    #         }

    #     return hyperedge_dict


    # **重点模块**
    def get_2hop_neighbor_linear_optimized(self, edge_index_dict):
        # 构造二跳邻接关系：给定一跳边索引，找到共享中间节点的二跳配对，
        # 以实现“中间节点作为桥接关系”的图结构扩展。
        def intersect_with_indices(a, b):
            a_sorted, idx_a_sorted = torch.sort(a)
            b_sorted, idx_b_sorted = torch.sort(b)

            pos = torch.searchsorted(b_sorted, a_sorted)

            valid = pos < b_sorted.numel()
            # 第一次：筛选（a -> b）边
            pos_valid = pos[valid]
            a_valid = a_sorted[valid]
            # 第二次：筛选（b -> a）边
            equal = b_sorted[pos_valid] == a_valid

            final = valid.clone()
            final[valid] = equal

            common = a_sorted[final]
            idx_a = idx_a_sorted[final]
            idx_b = idx_b_sorted[pos[final]]

            return common, idx_a, idx_b

        two_hop_neighbors = {}
        self.edge_search_dict = {(k[0], k[2]): k for k in self.edge_types}

        for (src, mid, dst), hop_type in self.two_hop_relations:

            if hop_type == 'co-occurrence':
                rel_l = self.edge_search_dict.get((mid, src))
            else:  # completion
                rel_l = self.edge_search_dict.get((src, mid))

            rel_r = self.edge_search_dict.get((mid, dst))

            if rel_l is None or rel_r is None:
                continue

            edge_l = edge_index_dict[rel_l]
            edge_r = edge_index_dict[rel_r]

            if hop_type == 'co-occurrence':
                mid_l, src_nodes = edge_l[0], edge_l[1]
            else:
                src_nodes, mid_l = edge_l[0], edge_l[1]

            mid_r, dst_nodes = edge_r[0], edge_r[1]

            common_mid, idx_l, idx_r = intersect_with_indices(
                mid_l, mid_r
            )

            if common_mid.numel() == 0:
                two_hop_neighbors[(src, mid, dst)] = (
                    torch.empty((2, 0), device=self.device, dtype=torch.long),
                    torch.empty((0,), device=self.device, dtype=torch.long)
                )
                continue

            src_aligned = src_nodes[idx_l]
            dst_aligned = dst_nodes[idx_r]

            edge_index = torch.stack([src_aligned, dst_aligned], dim=0)
            # 二跳边索引 [2, E_2hop]；每条二跳边对应一个桥接中间节点 ID [E_2hop]。

            two_hop_neighbors[(src, mid, dst)] = (
                edge_index.to(self.device),
                common_mid.to(self.device)
            )

        return two_hop_neighbors

class FD_Model(torch.nn.Module):
    """结构约束辅助模块，鼓励图中关系满足某种“特征差分”或“关系判别”约束。"""
    def __init__(self, edge_types, channels):
        super().__init__()
        self.edge_types = edge_types

        self.FD_rel = torch.nn.ModuleDict()
        self.FD_judge = torch.nn.ModuleDict()

        for rel in edge_types:
            key = tuple_2_str(rel)
            self.FD_rel[key] = FDSubspace(channels, channels // 4)
            self.FD_judge[key] = RelJudgeDiff(channels)
            
    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        perm: str = 'id'
    ) -> Dict[NodeType, Tensor]:
        loss_fd = []
        loss_fd_judge = []
        for rel in self.edge_types:
            src, edge, dst = rel
            key = tuple_2_str(rel)
            if edge_index_dict[rel].numel() == 0:
                continue
            if perm == 'id':
                # 按 [2, E] 的边索引取两端特征，得到一一对应的 [E, C] 节点对。
                loss_fd.append(self.FD_rel[key].loss(x_dict[src][edge_index_dict[rel][0]], x_dict[dst][edge_index_dict[rel][1]]))
            elif perm == 'random':
                loss_fd.append(
                    self.FD_rel[key].loss(
                        x_dict[src][edge_index_dict[rel][0]], 
                        x_dict[dst][edge_index_dict[rel][1][random.randint(0, len(edge_index_dict[rel][1]) - 1)]]
                    )
                )
            elif perm == 'mean':
                pass
            loss_fd_judge.append(self.FD_judge[key].loss(x_dict[src][edge_index_dict[rel][0]], x_dict[dst][edge_index_dict[rel][1]]))

        # 两项均为跨关系取均值后的标量损失。
        return torch.stack(loss_fd).mean(), torch.stack(loss_fd_judge).mean()