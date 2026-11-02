import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from model import FROG
from text_embedder import GloveTextEmbedding
from torch import Tensor
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from tqdm import tqdm

from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_link_train_table_input
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.loader import SparseTensor
from relbench.tasks import get_task

from utils import *
from graph import make_pkey_fkey_graph
from exp_model import FD_Model
from load_train_args import load_train_args

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-event")
parser.add_argument("--task", type=str, default="user-attendance")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=1000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--beta", type=float, default=1e-6)
parser.add_argument("--gamma", type=float, default=1e-1)
parser.add_argument("--MAE", type=float, default=0.95)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--optim", type=str, default="both")
parser.add_argument("--id_aware", action="store_false", default=True)
parser.add_argument("--vis", action="store_true", default=False)
parser.add_argument("--scheduler", action="store_true", default=False)
parser.add_argument("--fix", action="store_true", default=False)
parser.add_argument("--fix_attn", type=float, default=0.5)   # 0 is full node, 1 is full edge
parser.add_argument(
    "--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--load_para", action="store_true", default=False)
args = parser.parse_args()

if args.load_para:
    args = load_train_args(args)
    
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=torch.device("cpu")), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

debug_edge(data)
two_hop_rel = get_2hop_relation(get_relation(data.edge_types))
two_hop_neighbor = None

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_link_train_table_input(table, task)
    dst_nodes_dict[split] = table_input.dst_nodes
    loader_dict[split] = NeighborLoader(
        data,
        # num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers*2)],  # OOM
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(4)],  # 4 is enough for our datasets <- max{distanse(table1, table2)}
        time_attr="time",
        input_nodes=table_input.src_nodes,
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def train() -> tuple[float, float, float]:
    for p in model_FD.parameters():
        p.requires_grad = False
    for p in model.parameters():
        p.requires_grad = True

    model.train()
    model_FD.eval()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred, x_dict = model.forward_dst_readout(
            batch, task.src_entity_table, task.dst_entity_table
        )
        pred = pred.flatten()

        batch_size = batch[task.src_entity_table].batch_size

        # Get ground-truth
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Get target label
        target = torch.isin(
            batch[task.dst_entity_table].batch
            + batch_size * batch[task.dst_entity_table].n_id,
            src_batch + batch_size * dst_index,
        ).float()

        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss_fd, loss_fd_judge = model_FD(x_dict, batch.edge_index_dict)
        loss = loss + args.beta * loss_fd + args.gamma * loss_fd_judge
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break
    if args.scheduler:
        scheduler.step()
        
    return (loss_accum / count_accum, loss_fd.item(), loss_fd_judge.item())

def train_FD() -> None:
    for p in model_FD.parameters():
        p.requires_grad = True
    for p in model.parameters():
        p.requires_grad = False

    model_FD.train()
    model.eval()

    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)
        optimizer_FD.zero_grad()

        _, x_dict = model.forward_dst_readout(
            batch, task.src_entity_table, task.dst_entity_table
        )

        loss_fd, loss_fd_judge = model_FD(x_dict, batch.edge_index_dict)

        loss = args.beta / (args.gamma + 1e-6) * loss_fd + loss_fd_judge
        loss.backward()
        optimizer_FD.step()
        
        steps += 1
        if steps > args.max_steps_per_epoch:
            break

@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
    # for batch in loader:
        batch = batch.to(device)

        pred, x_dict = model.forward_dst_readout(
            batch, task.src_entity_table, task.dst_entity_table
        )
        pred = pred.flatten()

        batch_size = batch[task.src_entity_table].batch_size
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=pred.device)
        scores[
            batch[task.dst_entity_table].batch, batch[task.dst_entity_table].n_id
        ] = torch.sigmoid(pred)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)

    pred = torch.cat(pred_list, dim=0).cpu().numpy()

    return pred

model = FROG(
    data=data,
    two_hop_relations=(two_hop_rel, two_hop_neighbor),
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,
    aggr=args.aggr,
    norm="layer_norm",
    id_awareness=True,
    MAE = args.MAE,
    dropout = args.dropout,
    optim = args.optim,
    fix = args.fix,
    fix_attn = args.fix_attn
).to(device)
model_FD = FD_Model(edge_types=data.edge_types, channels=args.channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
optimizer_FD = torch.optim.Adam(model_FD.parameters(), lr=args.lr, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=1e-5
)
train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)


edge_attn = {}

state_dict = None
best_val_metric = 0
state_epoch = -1
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    train_FD()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    # print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")
    test_metrics = task.evaluate(test(loader_dict["test"]))
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}, Test metrics: {test_metrics}")

    if val_metrics[tune_metric] >= best_val_metric:
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())
        state_epoch = epoch
        edge_attn = copy.deepcopy(model.gnn.attn)

if args.vis:
    torch.save(state_dict, f"ckpt/model_{args.dataset}_{args.task}.pth")
    torch.save(edge_attn, f"ckpt/RDB_attn_{args.dataset}_{args.task}.pt")

print('load best model', state_epoch)
model.load_state_dict(state_dict)
model.gnn.attn = edge_attn
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")

print("Table as node/edge:")
print(model.gnn.attn)