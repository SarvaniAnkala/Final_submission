# """
# advanced_gcn_model.py
# ────────────────────────────────────────────────────────────────────────────
# Self-contained GCN/Transformer utility module for DocLayNet.

# • Feature extractor  : builds simple geometric &-id embeddings
# • Graph builder      : converts DocLayNet samples to PyG Data objects
# • TransformerGCN     : 2-layer GCN ⟶ Transformer encoder ⟶ node classifier
# • AdvancedTrainer    : generic train / validate loops
# • Data-loader helper : create_advanced_data_loaders()
# • Plotting utility   : plot_advanced_results()
# • UserConfig         : central hyper-parameter store
# ────────────────────────────────────────────────────────────────────────────
# Requires:
#     pip install torch torchvision torch_geometric scipy matplotlib scikit-learn
# """

# import os, math, random, json, itertools, warnings, time
# from typing import List, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# import numpy as np
# from scipy.spatial import cKDTree

# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader as GeoLoader
# from torch_geometric.nn import GCNConv
# from torch_geometric.utils import to_dense_batch

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# # ───────────────────────────── Hyper-parameters ──────────────────────────
# class UserConfig:
#     HIDDEN_DIMS        = 128          # node-hidden dim
#     NUM_HEADS          = 4            # transformer heads
#     TRANSFORMER_LAYERS = 2
#     DROPOUT            = 0.1
#     LEARNING_RATE      = 3e-4
#     WEIGHT_DECAY       = 1e-4
#     BATCH_SIZE         = 4            # graphs / mini-batch
#     EPOCHS             = 50
#     EARLY_STOP_PATIENCE= 6
#     K_NEIGHBORS        = 10           # KNN edges per node
#     MAX_NODES          = 300          # truncate very large pages


# # ────────────────────────── Feature extractor  ───────────────────────────
# class AdvancedFeatureExtractor(nn.Module):
#     """
#     Very light-weight MLP that embeds 9 geometry features + 1 class id
#     → HIDDEN_DIMS
#     Geometry features (all ∈ [0,1]):
#         x1, y1, x2, y2, w, h, cx, cy, area
#     """
#     def __init__(self, hidden: int, num_classes: int):
#         super().__init__()
#         self.embed_id = nn.Embedding(num_classes + 1, 16)       # id 0..n
#         self.mlp      = nn.Sequential(
#             nn.Linear(9 + 16, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden)
#         )

#     def forward(self, geom: torch.Tensor, labels: torch.Tensor):
#         """
#         geom   : (N,9)   float
#         labels : (N,)    int64
#         """
#         id_vec  = self.embed_id(labels)
#         x       = torch.cat([geom, id_vec], dim=-1)
#         return self.mlp(x)


# # ───────────────────────────── Graph builder ─────────────────────────────
# class GraphBuilder:
#     """
#     Convert single sample from DocLayNetDataset (see doclaynet_data_loader.py)
#     to torch_geometric.data.Data
#     """
#     def __init__(self, img_size=(1025,1025), k=10, max_nodes=300):
#         self.img_w, self.img_h = img_size
#         self.k  = k
#         self.max_nodes = max_nodes

#     def _norm(self, arr, div):
#         return arr.astype(np.float32) / float(div)

#     def build(self, sample: dict) -> Data:
#         """
#         sample dict keys:
#             boxes  : (M,4)  xyxy  float32
#             labels : (M,)   int64
#             image   not needed
#         """
#         boxes  = sample['boxes']
#         labels = sample['labels']
#         if boxes.shape[0] > self.max_nodes:                # clip long pages
#             keep = np.arange(boxes.shape[0])[:self.max_nodes]
#             boxes, labels = boxes[keep], labels[keep]

#         x1, y1, x2, y2 = boxes.T
#         w   = x2 - x1
#         h   = y2 - y1
#         cx  = (x1 + x2) / 2.0
#         cy  = (y1 + y2) / 2.0
#         area= w * h

#         geom = np.stack([
#             self._norm(x1, self.img_w),
#             self._norm(y1, self.img_h),
#             self._norm(x2, self.img_w),
#             self._norm(y2, self.img_h),
#             self._norm(w , self.img_w),
#             self._norm(h , self.img_h),
#             self._norm(cx, self.img_w),
#             self._norm(cy, self.img_h),
#             self._norm(area, self.img_w * self.img_h)
#         ], axis=1)

#         # KNN edges on centers
#         kd   = cKDTree(np.stack([cx, cy], axis=1))
#         _, knn = kd.query(np.stack([cx, cy], axis=1), k=min(self.k+1, len(cx)))
#         src = np.repeat(np.arange(len(cx)), knn.shape[1]-1)
#         dst = knn[:,1:].reshape(-1)
#         edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

#         data = Data(
#             pos      = torch.tensor(np.stack([cx, cy], axis=1), dtype=torch.float32),
#             geom     = torch.tensor(geom, dtype=torch.float32),
#             y        = torch.tensor(labels, dtype=torch.long),
#             edge_index = edge_index
#         )
#         return data


# # ─────────────────────────────  GCN backbone  ────────────────────────────
# class TransformerEnhancedGCN(nn.Module):
#     """
#     2-layer GCN ➜ TransformerEncoder ➜ FC head per node.
#     """
#     def __init__(self, num_features: int, num_classes: int, config: UserConfig):
#         super().__init__()
#         hidden = config.HIDDEN_DIMS
#         self.feat_extractor = AdvancedFeatureExtractor(hidden, num_classes)

#         self.gcn1  = GCNConv(hidden, hidden)
#         self.gcn2  = GCNConv(hidden, hidden)

#         enc_layer  = nn.TransformerEncoderLayer(
#                         d_model = hidden,
#                         nhead   = config.NUM_HEADS,
#                         dim_feedforward = hidden*2,
#                         dropout = config.DROPOUT,
#                         batch_first = True)
#         self.transformer = nn.TransformerEncoder(enc_layer,
#                                                  num_layers=config.TRANSFORMER_LAYERS)

#         self.head  = nn.Linear(hidden, num_classes)

#     def forward(self, data: Data):
#         # embed raw geometry + labels
#         x  = self.feat_extractor(data.geom, data.y.clamp(min=0))  # use true label ids
#         x  = F.relu(self.gcn1(x, data.edge_index))
#         x  = F.relu(self.gcn2(x, data.edge_index))

#         # transformer expects (B,L,H) dense batch
#         dense_x, mask = to_dense_batch(x, batch=data.batch)
#         x_trans = self.transformer(dense_x, src_key_padding_mask=~mask)
#         x = x_trans[mask]                                    # flatten back to (N,H)
#         out = self.head(x)
#         return out                                            # shape (N, num_classes)


# # ───────────────────────── Training / Validation Helper ──────────────────
# class AdvancedTrainer:
#     def __init__(self, model, device, cfg: UserConfig):
#         self.model = model
#         self.device= device
#         self.cfg   = cfg
#         self.optimizer = torch.optim.AdamW(model.parameters(),
#                                            lr=cfg.LEARNING_RATE,
#                                            weight_decay=cfg.WEIGHT_DECAY)
#         self.criterion = nn.CrossEntropyLoss()

#         self.train_losses, self.val_losses = [], []
#         self.train_accs , self.val_accs   = [], []

#     # ──────────────────────────────────────────────────────────────────────
#     def _step(self, loader, is_train=True):
#         mode = 'train' if is_train else 'eval'
#         getattr(self.model, mode)()

#         total_loss, correct, total = 0.0, 0, 0
#         preds_all, labels_all = [], []

#         for data in loader:
#             data = data.to(self.device)
#             if is_train:
#                 self.optimizer.zero_grad()

#             out   = self.model(data)
#             loss  = self.criterion(out, data.y)
#             if is_train:
#                 loss.backward()
#                 self.optimizer.step()

#             total_loss += loss.item() * data.num_nodes
#             pred  = out.argmax(dim=1)
#             correct += (pred == data.y).sum().item()
#             total   += data.num_nodes

#             preds_all.append(pred.cpu())
#             labels_all.append(data.y.cpu())

#         avg_loss = total_loss / total
#         acc      = correct   / total
#         preds_all= torch.cat(preds_all).tolist()
#         labels_all = torch.cat(labels_all).tolist()
#         return avg_loss, acc, preds_all, labels_all

#     # ──────────────────────────────────────────────────────────────────────
#     def train_epoch(self, loader):
#         loss, acc, _, _ = self._step(loader, is_train=True)
#         self.train_losses.append(loss)
#         self.train_accs.append(acc)
#         return loss, acc

#     def validate_epoch(self, loader):
#         loss, acc, preds, labels = self._step(loader, is_train=False)
#         self.val_losses.append(loss)
#         self.val_accs.append(acc)
#         return loss, acc, preds, labels


# # ─────────────────────── PyG Data-Loader Factory ─────────────────────────
# def create_advanced_data_loaders(dataset,
#                                  cfg: UserConfig = UserConfig(),
#                                  split_ratios=(0.8,0.1,0.1),
#                                  shuffle=True, seed=42):
#     """
#     dataset : instance of DocLayNetDataset (or similar)
#     Returns : train_loader, val_loader, test_loader, num_features
#     """
#     builder = GraphBuilder(k=cfg.K_NEIGHBORS, max_nodes=cfg.MAX_NODES)

#     # build all graphs first (takes ~2-3 s for 1 %)
#     graphs: List[Data] = []
#     for idx in range(len(dataset)):
#         samp = dataset[idx]
#         g    = builder.build({
#                 'boxes' : samp[1]['boxes'].numpy(),
#                 'labels': samp[1]['labels'].numpy()
#         })
#         g.batch = torch.zeros(g.num_nodes, dtype=torch.long)   # dummy batch idx
#         graphs.append(g)

#     if shuffle:
#         random.seed(seed)
#         random.shuffle(graphs)

#     n_total = len(graphs)
#     n_train = int(split_ratios[0]*n_total)
#     n_val   = int(split_ratios[1]*n_total)

#     train_set, val_set, test_set = (graphs[:n_train],
#                                     graphs[n_train:n_train+n_val],
#                                     graphs[n_train+n_val:])

#     train_loader = GeoLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
#     val_loader   = GeoLoader(val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False)
#     test_loader  = GeoLoader(test_set,  batch_size=cfg.BATCH_SIZE, shuffle=False)

#     num_features = cfg.HIDDEN_DIMS    # after feat extractor
#     return train_loader, val_loader, test_loader, num_features


# # ───────────────────────────── Plotting utils ────────────────────────────
# def plot_advanced_results(tr_losses, val_losses,
#                           tr_accs, val_accs,
#                           y_true, y_pred, class_names: List[str]):
#     epochs = list(range(1, len(tr_losses)+1))
#     fig, ax = plt.subplots(1,2, figsize=(12,5))

#     ax[0].plot(epochs, tr_losses, label='Train Loss')
#     ax[0].plot(epochs, val_losses, label='Val Loss')
#     ax[0].set_xlabel('Epoch'); ax[0].set_title('Loss'); ax[0].legend()

#     ax[1].plot(epochs, tr_accs, label='Train Acc')
#     ax[1].plot(epochs, val_accs, label='Val Acc')
#     ax[1].set_xlabel('Epoch'); ax[1].set_title('Accuracy'); ax[1].legend()

#     plt.tight_layout()
#     plt.savefig('training_curves.png')
#     print("→ Saved curves to training_curves.png")

#     # confusion matrix
#     cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
#     disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
#     fig2, ax2 = plt.subplots(figsize=(6,6))
#     disp.plot(ax=ax2, xticks_rotation=45, cmap='Blues')
#     plt.tight_layout()
#     plt.savefig('confusion_matrix.png')
#     print("→ Saved confusion matrix to confusion_matrix.png")




"""
advanced_gcn_model.py
────────────────────────────────────────────────────────────────────────────
Self-contained GCN/Transformer utility module for DocLayNet.

• Feature extractor  : builds simple geometric &-id embeddings
• Graph builder      : converts DocLayNet samples to PyG Data objects
• TransformerGCN     : 2-layer GCN ⟶ Transformer encoder ⟶ node classifier
• AdvancedTrainer    : generic train / validate loops
• Data-loader helper : create_advanced_data_loaders()
• Plotting utility   : plot_advanced_results()
• UserConfig         : central hyper-parameter store
────────────────────────────────────────────────────────────────────────────
Requires:
    pip install torch torchvision torch_geometric scipy matplotlib scikit-learn
"""

import os, math, random, json, itertools, warnings, time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from scipy.spatial import cKDTree

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ───────────────────────────── Hyper-parameters ──────────────────────────
class UserConfig:
    HIDDEN_DIMS        = 128          # node-hidden dim
    NUM_HEADS          = 4            # transformer heads
    TRANSFORMER_LAYERS = 2
    DROPOUT            = 0.1
    LEARNING_RATE      = 3e-4
    WEIGHT_DECAY       = 1e-4
    BATCH_SIZE         = 4            # graphs / mini-batch
    EPOCHS             = 50
    EARLY_STOP_PATIENCE= 6
    K_NEIGHBORS        = 10           # KNN edges per node
    MAX_NODES          = 300          # truncate very large pages

# ────────────────────────── Feature extractor  ───────────────────────────
class AdvancedFeatureExtractor(nn.Module):
    """
    Very light-weight MLP that embeds 9 geometry features + 1 class id
    → HIDDEN_DIMS
    Geometry features (all ∈ [0,1]):
        x1, y1, x2, y2, w, h, cx, cy, area
    """
    def __init__(self, hidden: int, num_classes: int):
        super().__init__()
        self.embed_id = nn.Embedding(num_classes + 1, 16)       # id 0..n
        self.mlp      = nn.Sequential(
            nn.Linear(9 + 16, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, geom: torch.Tensor, labels: torch.Tensor):
        """
        geom   : (N,9)   float
        labels : (N,)    int64
        """
        # Ensure labels are in valid range for embedding
        labels = labels.clamp(0, self.embed_id.num_embeddings - 1)
        id_vec  = self.embed_id(labels)
        x       = torch.cat([geom, id_vec], dim=-1)
        return self.mlp(x)

# ───────────────────────────── Graph builder ─────────────────────────────
class GraphBuilder:
    """
    Convert single sample from DocLayNetDataset (see doclaynet_data_loader.py)
    to torch_geometric.data.Data with robust error handling
    """
    def __init__(self, img_size=(1025,1025), k=10, max_nodes=300):
        self.img_w, self.img_h = img_size
        self.k  = k
        self.max_nodes = max_nodes

    def _norm(self, arr, div):
        return arr.astype(np.float32) / float(div)

    def build(self, sample: dict) -> Data:
        """
        sample dict keys:
            boxes  : (M,4)  xyxy  float32
            labels : (M,)   int64
            image   not needed
        """
        boxes  = sample['boxes']
        labels = sample['labels']
        
        # Handle empty documents
        if boxes.shape[0] == 0:
            print("Warning: Empty document found, creating dummy node")
            boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            labels = np.array([0], dtype=np.int64)
        
        # CRITICAL FIX: Ensure labels are in valid range [0, 10] for 11 classes
        original_min, original_max = labels.min(), labels.max()
        labels = np.clip(labels, 0, 10)
        
        if original_min < 0 or original_max > 10:
            print(f"Warning: Invalid labels detected - min: {original_min}, max: {original_max}")
            print(f"Clamped to valid range [0, 10]")
        
        # Clip very long pages
        original_length = boxes.shape[0]
        if boxes.shape[0] > self.max_nodes:
            keep = np.arange(boxes.shape[0])[:self.max_nodes]
            boxes, labels = boxes[keep], labels[keep]
            if original_length > self.max_nodes:
                print(f"Warning: Clipped document from {original_length} to {self.max_nodes} nodes")

        x1, y1, x2, y2 = boxes.T
        w   = x2 - x1
        h   = y2 - y1
        cx  = (x1 + x2) / 2.0
        cy  = (y1 + y2) / 2.0
        area= w * h

        geom = np.stack([
            self._norm(x1, self.img_w),
            self._norm(y1, self.img_h),
            self._norm(x2, self.img_w),
            self._norm(y2, self.img_h),
            self._norm(w , self.img_w),
            self._norm(h , self.img_h),
            self._norm(cx, self.img_w),
            self._norm(cy, self.img_h),
            self._norm(area, self.img_w * self.img_h)
        ], axis=1)

        # Build edges with robust KNN handling
        num_nodes = len(cx)
        
        try:
            if num_nodes == 1:
                # Single node case - create self-loop
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            elif num_nodes == 2:
                # Two nodes - connect them bidirectionally
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            else:
                # Multiple nodes - use KNN
                centers = np.stack([cx, cy], axis=1)
                kd = cKDTree(centers)
                
                k_actual = min(self.k + 1, num_nodes)
                distances, knn = kd.query(centers, k=k_actual)
                
                # Ensure knn is always 2D
                if knn.ndim == 1:
                    knn = knn.reshape(1, -1)
                
                # Build edge lists
                src_list = []
                dst_list = []
                
                for i in range(num_nodes):
                    # Get neighbors (excluding self)
                    if knn.shape[1] > 1:
                        neighbors = knn[i, 1:]  # Skip self-connection
                        for neighbor in neighbors:
                            if neighbor != i and neighbor < num_nodes:
                                src_list.append(i)
                                dst_list.append(neighbor)
                
                # Fallback: if no valid edges, create a simple chain
                if len(src_list) == 0:
                    for i in range(num_nodes - 1):
                        src_list.extend([i, i+1])
                        dst_list.extend([i+1, i])
                
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        
        except Exception as e:
            print(f"Warning: KNN failed ({e}), using simple chain connectivity")
            # Fallback: create simple chain connectivity
            if num_nodes == 1:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            else:
                src_list = list(range(num_nodes - 1)) + list(range(1, num_nodes))
                dst_list = list(range(1, num_nodes)) + list(range(num_nodes - 1))
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        data = Data(
            pos      = torch.tensor(np.stack([cx, cy], axis=1), dtype=torch.float32),
            geom     = torch.tensor(geom, dtype=torch.float32),
            y        = torch.tensor(labels, dtype=torch.long),  # Now guaranteed to be [0, 10]
            edge_index = edge_index
        )
        return data

# ─────────────────────────────  GCN backbone  ────────────────────────────
class TransformerEnhancedGCN(nn.Module):
    """
    2-layer GCN ➜ TransformerEncoder ➜ FC head per node.
    """
    def __init__(self, num_features: int, num_classes: int, config: UserConfig):
        super().__init__()
        hidden = config.HIDDEN_DIMS
        self.num_classes = num_classes
        self.feat_extractor = AdvancedFeatureExtractor(hidden, num_classes)

        self.gcn1  = GCNConv(hidden, hidden)
        self.gcn2  = GCNConv(hidden, hidden)

        enc_layer  = nn.TransformerEncoderLayer(
                        d_model = hidden,
                        nhead   = config.NUM_HEADS,
                        dim_feedforward = hidden*2,
                        dropout = config.DROPOUT,
                        batch_first = True)
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                 num_layers=config.TRANSFORMER_LAYERS)

        self.head  = nn.Linear(hidden, num_classes)

    def forward(self, data: Data):
        # CRITICAL FIX: Ensure labels are in valid range before processing
        data.y = data.y.clamp(0, self.num_classes - 1)
        
        # embed raw geometry + labels
        x  = self.feat_extractor(data.geom, data.y)  # Labels already clamped
        x  = F.relu(self.gcn1(x, data.edge_index))
        x  = F.relu(self.gcn2(x, data.edge_index))

        # transformer expects (B,L,H) dense batch
        dense_x, mask = to_dense_batch(x, batch=data.batch)
        x_trans = self.transformer(dense_x, src_key_padding_mask=~mask)
        x = x_trans[mask]                                    # flatten back to (N,H)
        out = self.head(x)
        return out                                            # shape (N, num_classes)

# ───────────────────────── Training / Validation Helper ──────────────────
class AdvancedTrainer:
    def __init__(self, model, device, cfg: UserConfig):
        self.model = model
        self.device= device
        self.cfg   = cfg
        self.optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=cfg.LEARNING_RATE,
                                           weight_decay=cfg.WEIGHT_DECAY)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses, self.val_losses = [], []
        self.train_accs , self.val_accs   = [], []

    # ──────────────────────────────────────────────────────────────────────
    def _step(self, loader, is_train=True):
        mode = 'train' if is_train else 'eval'
        getattr(self.model, mode)()

        total_loss, correct, total = 0.0, 0, 0
        preds_all, labels_all = [], []

        for batch_idx, data in enumerate(loader):
            data = data.to(self.device)
            
            # CRITICAL FIX: Validate and clamp labels before loss computation
            original_labels = data.y.clone()
            data.y = data.y.clamp(0, 10)  # Force labels to [0, 10] for 11 classes
            
            # Debug: Check for problematic labels
            if torch.any(original_labels != data.y):
                invalid_count = torch.sum(original_labels != data.y).item()
                print(f"Batch {batch_idx}: Fixed {invalid_count} invalid labels")
                print(f"  Original range: [{original_labels.min().item()}, {original_labels.max().item()}]")
                print(f"  Clamped range: [{data.y.min().item()}, {data.y.max().item()}]")
            
            if is_train:
                self.optimizer.zero_grad()

            try:
                out   = self.model(data)
                loss  = self.criterion(out, data.y)  # Now using clamped labels
                
                if is_train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * data.num_nodes
                pred  = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total   += data.num_nodes

                preds_all.append(pred.cpu())
                labels_all.append(data.y.cpu())
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Data.y range: [{data.y.min().item()}, {data.y.max().item()}]")
                print(f"Num nodes: {data.num_nodes}")
                raise e

        avg_loss = total_loss / total if total > 0 else 0.0
        acc      = correct   / total if total > 0 else 0.0
        preds_all= torch.cat(preds_all).tolist() if preds_all else []
        labels_all = torch.cat(labels_all).tolist() if labels_all else []
        return avg_loss, acc, preds_all, labels_all

    # ──────────────────────────────────────────────────────────────────────
    def train_epoch(self, loader):
        loss, acc, _, _ = self._step(loader, is_train=True)
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        return loss, acc

    def validate_epoch(self, loader):
        loss, acc, preds, labels = self._step(loader, is_train=False)
        self.val_losses.append(loss)
        self.val_accs.append(acc)
        return loss, acc, preds, labels

# ─────────────────────── PyG Data-Loader Factory ─────────────────────────
def create_advanced_data_loaders(dataset,
                                 cfg: UserConfig = UserConfig(),
                                 split_ratios=(0.8,0.1,0.1),
                                 shuffle=True, seed=42):
    """
    dataset : instance of DocLayNetDataset (or similar)
    Returns : train_loader, val_loader, test_loader, num_features
    """
    builder = GraphBuilder(k=cfg.K_NEIGHBORS, max_nodes=cfg.MAX_NODES)

    # build all graphs first (takes ~2-3 s for 1 %)
    graphs: List[Data] = []
    print(f"Converting {len(dataset)} samples to graphs...")
    
    failed_conversions = 0
    label_stats = {"min": float('inf'), "max": float('-inf'), "invalid_count": 0}
    
    for idx in range(len(dataset)):
        try:
            samp = dataset[idx]
            
            # Check labels before conversion
            raw_labels = samp[1]['labels'].numpy()
            if raw_labels.min() < 0 or raw_labels.max() > 10:
                label_stats["invalid_count"] += 1
                label_stats["min"] = min(label_stats["min"], raw_labels.min())
                label_stats["max"] = max(label_stats["max"], raw_labels.max())
            
            g = builder.build({
                'boxes' : samp[1]['boxes'].numpy(),
                'labels': raw_labels
            })
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long)   # dummy batch idx
            graphs.append(g)
            
        except Exception as e:
            failed_conversions += 1
            print(f"Warning: Failed to convert sample {idx}: {e}")
            continue
    
    if failed_conversions > 0:
        print(f"Warning: {failed_conversions} samples failed conversion")
    
    if label_stats["invalid_count"] > 0:
        print(f"Label validation: {label_stats['invalid_count']} samples had invalid labels")
        print(f"  Label range found: [{label_stats['min']}, {label_stats['max']}]")
        print(f"  Valid range: [0, 10]")
    
    if len(graphs) == 0:
        raise ValueError("No graphs were successfully created!")

    if shuffle:
        random.seed(seed)
        random.shuffle(graphs)

    n_total = len(graphs)
    n_train = int(split_ratios[0]*n_total)
    n_val   = int(split_ratios[1]*n_total)

    train_set, val_set, test_set = (graphs[:n_train],
                                    graphs[n_train:n_train+n_val],
                                    graphs[n_train+n_val:])

    print(f"Created {len(train_set)} train, {len(val_set)} val, {len(test_set)} test graphs")

    train_loader = GeoLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = GeoLoader(val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader  = GeoLoader(test_set,  batch_size=cfg.BATCH_SIZE, shuffle=False)

    num_features = cfg.HIDDEN_DIMS    # after feat extractor
    return train_loader, val_loader, test_loader, num_features

# ───────────────────────────── Plotting utils ────────────────────────────
def plot_advanced_results(tr_losses, val_losses,
                          tr_accs, val_accs,
                          y_true, y_pred, class_names: List[str],
                          title_prefix="TransformerEnhanced GCN"):
    """
    Plot training curves and confusion matrix
    """
    if len(tr_losses) == 0:
        print("Warning: No training data to plot")
        return
        
    epochs = list(range(1, len(tr_losses)+1))
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    ax[0].plot(epochs, tr_losses, label='Train Loss', marker='o')
    ax[0].plot(epochs, val_losses, label='Val Loss', marker='s')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title(f'{title_prefix} - Loss')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs, tr_accs, label='Train Acc', marker='o')
    ax[1].plot(epochs, val_accs, label='Val Acc', marker='s')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title(f'{title_prefix} - Accuracy')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("→ Saved curves to training_curves.png")

    # confusion matrix
    if len(y_true) > 0 and len(y_pred) > 0 and len(set(y_true)) > 1:
        try:
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            fig2, ax2 = plt.subplots(figsize=(10,8))
            disp.plot(ax=ax2, xticks_rotation=45, cmap='Blues', values_format='d')
            plt.title(f'{title_prefix} - Confusion Matrix')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
            print("→ Saved confusion matrix to confusion_matrix.png")
        except Exception as e:
            print(f"Warning: Could not create confusion matrix: {e}")
    else:
        print("Warning: Insufficient data for confusion matrix")

# Add alias for backward compatibility
AdvancedGCNModel = TransformerEnhancedGCN
