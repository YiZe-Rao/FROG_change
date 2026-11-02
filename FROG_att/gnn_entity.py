# ============================================================
# FROG entity training entry script
# 作用：读取 RelBench 数据集/任务，构建异构图，训练 FROG 模型，
# 并在验证/测试集上做实体分类、回归或多分类任务评估。
# ============================================================
import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from model import FROG
from sklearn.preprocessing import LabelEncoder
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, Table, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task, get_task_names

from utils import *
from graph import make_pkey_fkey_graph
from exp_model import FD_Model
from load_train_args import load_train_args

# ------------------------------------------------------------------
# 1. 训练参数配置
# 作用：定义数据集、任务、模型超参和训练策略，可在命令行中覆盖。
# 例如 dataset/task 决定训练哪一个关系数据集及其目标任务。
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-event")
parser.add_argument("--task", type=str, default="user-attendance")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--warmup", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--beta", type=float, default=1e-6)
parser.add_argument("--gamma", type=float, default=1e-1)
parser.add_argument("--MAE", type=float, default=0.99)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--optim", type=str, default="both")
parser.add_argument("--id_aware", action="store_false", default=True)
parser.add_argument("--vis", action="store_true", default=False)
parser.add_argument("--scheduler", action="store_true", default=False)
parser.add_argument("--fix", action="store_true", default=False)
parser.add_argument("--fix_attn", type=float, default=0.5)   # 0 is full node, 1 is full edge, -1 is random
parser.add_argument(
    "--include_task_tables",
    type=str,
    default="none",
    help="Optionally include labels as autoregressive features with \
        appropriate time censoring. One of 'all', \
        'current_only', or 'none'.",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--load_para", action="store_true", default=False)

# 解析命令行参数，并在需要时载入针对特定 dataset/task 的推荐超参。
args = parser.parse_args()

if args.load_para:
    args = load_train_args(args)

print(args)

# 选择设备：优先 GPU，否则 CPU；并固定随机种子，保证实验可复现。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

# 加载数据集和当前任务，dataset 是整个数据库，task 是具体目标任务。
dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)


# ------------------------------------------------------------------
# 2. 列类型推断与缓存
# 作用：读取/生成每一列的类型（数值、文本、时间、类别等），
# 这是 Torch Frame 建模所需的 schema 信息，可缓存减少重复推断。
# ------------------------------------------------------------------
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

# 可选：将其他任务的标签表也加入数据库，作为时间截断后的自回归特征。
if args.include_task_tables == "all":
    tasks_to_add = get_task_names(args.dataset)
elif args.include_task_tables == "current_only":
    tasks_to_add = [args.task]
else:
    tasks_to_add = []

db = dataset.get_db()

# ------------------------------------------------------------------
# 3. 构建时序标签表并生成异构图
# 作用：把关系数据库转成可用于 GNN 的异构图数据结构，
# 同时加入时间截断后的标签表，以避免数据泄漏。
# ------------------------------------------------------------------
for task_name in tasks_to_add:
    t = get_task(args.dataset, task_name)
    if not isinstance(t, EntityTask):
        continue
    labels_table_name = f"{task_name}_labels"
    label_df = pd.concat(
        [
            t.get_table("train").df,
            t.get_table("val").df,
            # test set not included b/c labels are not revealed
        ]
    )
    # time-censoring labels: we add timedelta to the time column to ensure that
    # the labels become available at the appropriate time (i.e. no leakage)
    label_df[t.time_col] = label_df[t.time_col] + t.timedelta
    db.table_dict[labels_table_name] = Table(
        df=label_df,
        fkey_col_to_pkey_table={t.entity_col: t.entity_table},
        pkey_col=None,
        time_col=t.time_col,
    )
    col_to_stype_dict[labels_table_name] = {
        t.entity_col: stype.numerical,
        t.time_col: stype.timestamp,
        t.target_col: stype.numerical,
    }

cache_name = (
    args.include_task_tables
    if args.include_task_tables != "current_only"
    else args.task
)
data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=torch.device("cpu")), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}_{cache_name}/materialized",
)

debug_edge(data)
# 获取二跳关系，用于 FROG 中的结构学习增强：除了一跳图边，还建模二跳交互。
two_hop_rel = get_2hop_relation(get_relation(data.edge_types))
two_hop_neighbor = None

# ------------------------------------------------------------------
# 4. 根据任务类型设定输出头和损失函数
# 作用：不同任务需要不同的目标输出维度和评估指标。
# 例如：二分类 → BCE + ROC-AUC；回归 → L1 + MAE；多分类 → CrossEntropy。
# ------------------------------------------------------------------
clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
    out_channels = task.num_classes
    loss_fn = CrossEntropyLoss()
    tune_metric = "multiclass_f1"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

# ------------------------------------------------------------------
# 5. 构造训练/验证/测试的邻居采样加载器
# 作用：NeighborLoader 负责按时间顺序抽取局部子图，使模型能在节点级任务中
# 获取每个目标节点周围的异构邻居信息，避免一次性加载全图。
# ------------------------------------------------------------------
loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    loader_dict[split] = NeighborLoader(
        data,
        # num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers * 2)],
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(max(3, args.num_layers * 2))], # enough neighbor info
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
# 每个 batch 是按时间采样的局部异构子图：各类型节点特征为 [N_t, F_t]，
# edge_index_dict[关系] 为 [2, E]；目标实体的前 B 个节点对应本批监督样本。


# ------------------------------------------------------------------
# 6. 训练函数：主模型更新
# 作用：在每个 epoch 中训练主网络，使用任务损失 + 结构学习辅助损失。
# 其中 model_FD 是一个辅助模块，用于进一步约束图边的结构判断。
# ------------------------------------------------------------------
def train() -> tuple[float, float, float]:
    # 先冻结辅助模块，更新主 FROG 模型；后续再单独训练辅助模块。
    for p in model_FD.parameters():
        p.requires_grad = False
    for p in model.parameters():
        p.requires_grad = True

    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)

        optimizer.zero_grad()
        # pred: [B, out_channels]；x_dict[t]: [N_t, channels]，含采样子图中该类型全部节点的隐向量。
        pred, x_dict = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # 多分类标签 [B]；单输出 pred 已展平为 [B]，多标签 pred/y 保持 [B, num_labels]。
            loss = loss_fn(pred, batch[entity_table].y.long())
        else:
            loss = loss_fn(pred.float(), batch[entity_table].y.float())

        # 辅助结构损失：鼓励模型学习更合理的图关系与特征分布。
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

# ------------------------------------------------------------------
# 7. 训练函数：结构辅助模块更新
# 作用：单独训练 model_FD，让它专门优化图结构判断/关系约束，
# 以帮助主模型更稳定地学习关系结构。
# ------------------------------------------------------------------
def train_FD() -> None:
    for p in model_FD.parameters():
        p.requires_grad = True
    for p in model.parameters():
        p.requires_grad = False

    model_FD.train()

    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)
        optimizer_FD.zero_grad()

        _, x_dict = model(
            batch,
            task.entity_table,
        )

        loss_fd, loss_fd_judge = model_FD(x_dict, batch.edge_index_dict)

        loss = args.beta / (args.gamma + 1e-6) * loss_fd + loss_fd_judge
        loss.backward()
        optimizer_FD.step()
        
        steps += 1
        if steps > args.max_steps_per_epoch:
            break



# ------------------------------------------------------------------
# 8. 推理函数：验证/测试时调用
# 作用：在 eval 模式下对当前批次进行前向推理，并按任务类型转换概率/值的范围。
# 例如回归时裁剪到训练分位数范围，二分类时做 sigmoid，分类任务做 softmax。
# ------------------------------------------------------------------
@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred, _ = model(
            batch,
            task.entity_table,
        )
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = torch.softmax(pred, dim=1)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()

# ------------------------------------------------------------------
# 9. 模型初始化与优化器设置
# 作用：实例化主模型和辅助结构模型，并绑定优化器、学习率调度器。
# 这里的 FROG 会接收异构图数据和二跳关系信息，用于关系结构学习。
# ------------------------------------------------------------------
model = FROG(
    data=data,
    two_hop_relations=(two_hop_rel, two_hop_neighbor),
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
    id_awareness = args.id_aware,
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

edge_attn = {}

# ------------------------------------------------------------------
# 10. 训练主循环
# 作用：反复训练、验证和测试，记录最优验证指标，保留最佳模型参数。
# 这里的 edge_attn 会记录关系的重要性权重，便于分析“表作为节点/边”的相对贡献。
# ------------------------------------------------------------------
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
state_epoch = -1
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    train_FD()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    # print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")
    test_metrics = task.evaluate(test(loader_dict["test"]))
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}, Test metrics: {test_metrics}")

    if epoch > args.warmup:
        if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] <= best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())
            state_epoch = epoch
            edge_attn = copy.deepcopy(model.gnn.attn)

if args.vis:
    torch.save(state_dict, f"ckpt/model_{args.dataset}_{args.task}.pth")
    torch.save(edge_attn, f"ckpt/RDB_attn_{args.dataset}_{args.task}.pt")

# 载入最佳验证集模型并输出最终评估结果。
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