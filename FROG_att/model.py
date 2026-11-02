# ============================================================
# FROG 主模型定义
# 作用：将表征编码、时间编码、异构图消息传递和输出头组合在一起，
# 形成最终的实体级预测模型。
# ============================================================
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from exp_model import HeteroGraphSAGE


class FROG(torch.nn.Module):
    """FROG的主网络：特征编码 + 时间编码 + 关系图结构学习 + 预测头。"""

    def __init__(
        self,
        data: HeteroData,
        two_hop_relations: set,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
        MAE: float = 0.99,
        dropout: float = 0.0,
        optim: str = 'both',
        fix: bool = False,
        fix_attn: float = 0.5
    ):
        super().__init__()

        # 1. 处理原始表征：把每种节点类型的列特征编码成隐藏表示。
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        # 2. 处理时间特征：把时间信息融入节点表示，适用于时序关系数据。
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )

        # 3. 主图网络：在异构图上做消息传递，并融合 1-hop / 2-hop 关系。
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            two_hop_relations=two_hop_relations,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
            MAE = MAE,
            dropout=dropout,
            optim = optim,
            fix = fix,
            fix_attn = fix_attn
        )
        # 4. 输出头：将最终节点表示映射到任务输出空间。
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        # 5. 可选的浅层 embedding：常用于给某些节点类型额外一个 ID 级表征。
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化各个子模块参数，保证训练稳定。
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        # batch[entity_table].seed_time 表示目标节点对应的时间，
        # 用于时间感知采样和输出截断。
        seed_time = batch[entity_table].seed_time

        # 先从原始表特征生成节点表征。
        x_dict = self.encoder(batch.tf_dict)

        # 再加入时序编码（时间感知）
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        # 可选浅层 embedding 注入
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        # 通过异构图网络进行消息传递，融合关系结构信息。
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict
        )

        # 只取当前 batch 中 seed nodes 对应的输出作为预测结果。
        return self.head(x_dict[entity_table][: seed_time.size(0)]), x_dict

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        # 这个方法用于“读出目标表的表示”，通常用于推荐/链接任务。
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        # 给根节点增加 ID 相关的感知表征，增强目标真实身份的区分。
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table]), x_dict
