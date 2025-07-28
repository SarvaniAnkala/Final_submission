"""
Fixed GCN/Transformer model for DocLayNet with robust graph building.

Key fixes:
- Proper handling of single node cases
- Robust KNN edge creation with fallback strategies
- Better error handling and validation
- Memory efficient processing
- Fixed transformer masking
- Proper batch handling
- Gradient clipping and stability improvements
"""

import os, math, random, json, itertools, warnings, time
from typing import List, Tuple, Optional, Dict, Any
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch, add_self_loops, remove_self_loops

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# ───────────────────────── Hyper-parameters ──────────────────────────
class UserConfig:
    HIDDEN_DIMS        = 128          # node-hidden dim
    NUM_HEADS          = 4            # transformer heads
    TRANSFORMER_LAYERS = 2
    DROPOUT            = 0.15         # Slightly increased for better regularization
    LEARNING_RATE      = 5e-4         # More conservative learning rate
    WEIGHT_DECAY       = 1e-4
    BATCH_SIZE         = 8            
    EPOCHS             = 50           
    EARLY_STOP_PATIENCE= 15           # Increased patience
    K_NEIGHBORS        = 6            # Reduced for more stable graphs
    MAX_NODES          = 256          # Reduced for memory efficiency
    MIN_NODES          = 1            # Minimum nodes required


# ────────────────────────── Feature extractor  ───────────────────────────
class AdvancedFeatureExtractor(nn.Module):
    """
    Enhanced feature extractor with better normalization and regularization.
    Embeds geometry features (9) without using labels as input features → HIDDEN_DIMS
    """
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(9)
        
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden // 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden)
        )
        
        self.output_norm = nn.LayerNorm(hidden)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, geom: torch.Tensor):
        """
        geom : (N,9) float - normalized geometry features
        """
        # Input normalization (handle single sample case)
        if geom.size(0) > 1:
            geom = self.input_norm(geom)
        
        x = self.mlp(geom)
        return self.output_norm(x)


# ───────────────────────────── Graph builder ─────────────────────────────
class GraphBuilder:
    """
    Convert single sample from DocLayNetDataset to torch_geometric.data.Data
    with robust edge creation strategies.
    """
    def __init__(self, img_size=(1025, 1025), k=6, max_nodes=256, min_nodes=1):
        self.img_w, self.img_h = img_size
        self.k = k
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes

    def _normalize_coords(self, coords, max_val):
        """Safely normalize coordinates with clipping"""
        coords = np.asarray(coords, dtype=np.float32)
        max_val = float(max_val)
        return np.clip(coords / max_val, 0.0, 1.0)

    def _create_edges_knn(self, centers: np.ndarray, k: int) -> np.ndarray:
        """Create KNN edges with multiple fallback strategies"""
        n_nodes = len(centers)
        
        if n_nodes == 1:
            # Single node - create self-loop
            return np.array([[0], [0]])
        
        # Strategy 1: KNN with cKDTree
        try:
            kd = cKDTree(centers)
            k_actual = min(k + 1, n_nodes)  # +1 because first neighbor is self
            distances, knn_indices = kd.query(centers, k=k_actual)
            
            # Handle 1D case
            if knn_indices.ndim == 1:
                knn_indices = knn_indices.reshape(1, -1)
            
            src_nodes, dst_nodes = [], []
            for i in range(n_nodes):
                neighbors = knn_indices[i]
                # Skip self if we have other neighbors
                start_idx = 1 if len(neighbors) > 1 else 0
                for j in range(start_idx, len(neighbors)):
                    if neighbors[j] != i or len(neighbors) == 1:  # Avoid self-loops unless necessary
                        src_nodes.append(i)
                        dst_nodes.append(neighbors[j])
            
            if src_nodes:
                return np.array([src_nodes, dst_nodes])
        
        except Exception as e:
            warnings.warn(f"KNN strategy failed: {e}")
        
        # Strategy 2: Distance-based threshold
        try:
            distances = squareform(pdist(centers))
            threshold = np.percentile(distances[distances > 0], 20)  # Connect closest 20% pairs
            
            src_nodes, dst_nodes = [], []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and distances[i, j] <= threshold:
                        src_nodes.append(i)
                        dst_nodes.append(j)
            
            if src_nodes:
                return np.array([src_nodes, dst_nodes])
        
        except Exception as e:
            warnings.warn(f"Distance threshold strategy failed: {e}")
        
        # Strategy 3: Fully connected for small graphs
        if n_nodes <= 10:
            try:
                src_nodes, dst_nodes = [], []
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j:
                            src_nodes.append(i)
                            dst_nodes.append(j)
                
                if src_nodes:
                    return np.array([src_nodes, dst_nodes])
            
            except Exception as e:
                warnings.warn(f"Fully connected strategy failed: {e}")
        
        # Strategy 4: Sequential connections (fallback)
        try:
            src_nodes, dst_nodes = [], []
            for i in range(n_nodes - 1):
                src_nodes.extend([i, i + 1])
                dst_nodes.extend([i + 1, i])
            
            # Connect last to first for cycle
            if n_nodes > 2:
                src_nodes.extend([n_nodes - 1, 0])
                dst_nodes.extend([0, n_nodes - 1])
            
            if src_nodes:
                return np.array([src_nodes, dst_nodes])
        
        except Exception as e:
            warnings.warn(f"Sequential strategy failed: {e}")
        
        # Final fallback: self-loops for all nodes
        return np.array([list(range(n_nodes)), list(range(n_nodes))])

    def build(self, sample: dict) -> Optional[Data]:
        """
        Build graph from sample with robust error handling.
        
        sample dict keys:
            boxes  : (M,4)  xyxy  float32
            labels : (M,)   int64
        """
        try:
            boxes = sample['boxes']
            labels = sample['labels']
            
            # Validation
            if len(boxes) == 0:
                warnings.warn("Empty boxes in sample")
                return None
            
            if len(boxes) < self.min_nodes:
                warnings.warn(f"Too few nodes: {len(boxes)} < {self.min_nodes}")
                return None
                
            # Truncate if too many nodes
            if boxes.shape[0] > self.max_nodes:
                indices = np.random.choice(boxes.shape[0], self.max_nodes, replace=False)
                boxes = boxes[indices]
                labels = labels[indices]

            # Fix label indexing - convert to 0-based indexing
            labels = np.asarray(labels)
            unique_labels = np.unique(labels)
            
            # Check if labels are 1-based (common in COCO format)
            if unique_labels.min() > 0:
                print(f"Warning: Converting 1-based labels to 0-based. Original range: {unique_labels.min()}-{unique_labels.max()}")
                labels = labels - 1  # Convert to 0-based
                
            # Ensure all labels are within valid range
            labels = np.clip(labels, 0, 10)  # Clip to 0-10 for 11 classes

            # Extract coordinates
            x1, y1, x2, y2 = boxes.T
            
            # Ensure valid boxes
            w = np.maximum(x2 - x1, 1.0)
            h = np.maximum(y2 - y1, 1.0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = w * h

            # Normalize all features to [0,1]
            geom = np.stack([
                self._normalize_coords(x1, self.img_w),
                self._normalize_coords(y1, self.img_h),
                self._normalize_coords(x2, self.img_w),
                self._normalize_coords(y2, self.img_h),
                self._normalize_coords(w, self.img_w),
                self._normalize_coords(h, self.img_h),
                self._normalize_coords(cx, self.img_w),
                self._normalize_coords(cy, self.img_h),
                self._normalize_coords(area, self.img_w * self.img_h)
            ], axis=1)

            # Create edges using robust strategy
            centers = np.stack([cx, cy], axis=1)
            edge_array = self._create_edges_knn(centers, self.k)
            edge_index = torch.tensor(edge_array, dtype=torch.long)

            # Validate edge_index
            if edge_index.size(1) == 0:
                warnings.warn("No edges created, adding self-loops")
                n_nodes = len(labels)
                edge_index = torch.tensor([list(range(n_nodes)), list(range(n_nodes))], dtype=torch.long)

            # Create PyG Data object
            data = Data(
                x=torch.tensor(geom, dtype=torch.float32),
                pos=torch.tensor(centers, dtype=torch.float32),
                y=torch.tensor(labels, dtype=torch.long),
                edge_index=edge_index,
                num_nodes=len(labels)
            )
            
            # Validate the data object
            data.validate(raise_on_error=False)
            
            return data
            
        except Exception as e:
            warnings.warn(f"Error building graph: {e}")
            return None


# ─────────────────────────────  GCN backbone  ────────────────────────────
class TransformerEnhancedGCN(nn.Module):
    """
    Robust GCN + Transformer architecture with improved stability.
    """
    def __init__(self, num_features: int, num_classes: int, config: UserConfig):
        super().__init__()
        hidden = config.HIDDEN_DIMS
        self.num_classes = num_classes
        self.config = config
        
        # Feature extractor (no label leakage)
        self.feat_extractor = AdvancedFeatureExtractor(hidden, config.DROPOUT)

        # GCN layers with improved architecture
        self.gcn1 = GCNConv(hidden, hidden, improved=True, add_self_loops=True)
        self.gcn2 = GCNConv(hidden, hidden, improved=True, add_self_loops=True)
        self.gcn3 = GCNConv(hidden, hidden, improved=True, add_self_loops=True)
        
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(3)
        ])
        
        self.dropout = nn.Dropout(config.DROPOUT)

        # Transformer encoder with proper configuration
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.NUM_HEADS,
            dim_feedforward=hidden * 4,  # Increased feedforward dimension
            dropout=config.DROPOUT,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, 
            num_layers=config.TRANSFORMER_LAYERS
        )

        # Enhanced classification head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            
            nn.Linear(hidden // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, data: Data):
        try:
            # Extract features from geometry only
            x = self.feat_extractor(data.x)
            
            # Add self-loops to edge_index if not present
            edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
            
            # GCN layers with residual connections
            x_orig = x
            
            # First GCN layer
            x1 = self.gcn1(x, edge_index)
            x1 = self.gcn_norms[0](x1)
            x1 = F.relu(x1)
            x1 = self.dropout(x1)
            
            # Second GCN layer
            x2 = self.gcn2(x1, edge_index)
            x2 = self.gcn_norms[1](x2)
            x2 = F.relu(x2)
            x2 = self.dropout(x2)
            
            # Third GCN layer with residual connection
            x3 = self.gcn3(x2, edge_index)
            x3 = self.gcn_norms[2](x3)
            x3 = F.relu(x3)
            x3 = self.dropout(x3)
            
            # Residual connection
            x = x_orig + x3

            # Convert to dense batch for transformer
            dense_x, mask = to_dense_batch(x, batch=data.batch)
            
            # Handle single sample case
            if dense_x.size(0) == 1 and dense_x.size(1) == 1:
                # Skip transformer for single node
                x_trans = dense_x
            else:
                # Apply transformer with proper masking
                # Create attention mask (True for valid positions)
                attn_mask = mask if mask is not None else torch.ones(dense_x.size()[:2], dtype=torch.bool, device=dense_x.device)
                
                # Transformer expects src_key_padding_mask where True means ignore
                padding_mask = ~attn_mask if attn_mask is not None else None
                
                try:
                    x_trans = self.transformer(dense_x, src_key_padding_mask=padding_mask)
                except Exception as e:
                    warnings.warn(f"Transformer forward failed: {e}, using input")
                    x_trans = dense_x
            
            # Convert back to node-level features
            if mask is not None:
                x = x_trans[mask]
            else:
                x = x_trans.view(-1, x_trans.size(-1))
            
            # Final classification
            out = self.head(x)
            return out
            
        except Exception as e:
            warnings.warn(f"Forward pass error: {e}")
            # Fallback: return random predictions
            batch_size = data.x.size(0)
            return torch.randn(batch_size, self.num_classes, device=data.x.device)


# ───────────────────────── Training / Validation Helper ──────────────────
class AdvancedTrainer:
    """
    Enhanced trainer with better stability and error handling.
    """

    def __init__(self, model: nn.Module, device: torch.device, cfg: UserConfig):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg

        # Enhanced optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler (version-safe)
        from inspect import signature
        sched_kwargs = dict(
            optimizer=self.optimizer,
            mode="max",
            factor=0.7,
            patience=5,
            min_lr=1e-6
        )
        # Add verbose only if supported
        if "verbose" in signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
            sched_kwargs["verbose"] = True
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**sched_kwargs)

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Metric history
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def _loop(self, loader, train: bool):
        """Enhanced training/validation loop with better error handling"""
        self.model.train(mode=train)
        total_loss, correct, total = 0.0, 0, 0
        preds_all, labels_all = [], []
        
        failed_batches = 0
        successful_batches = 0

        for batch_idx, batch in enumerate(loader):
            try:
                data = batch.to(self.device)
                
                if train:
                    self.optimizer.zero_grad()

                # Forward pass
                out = self.model(data)
                
                # Validate shapes and labels
                if out.size(0) != data.y.size(0):
                    warnings.warn(f"Shape mismatch: out {out.shape}, y {data.y.shape}")
                    continue
                
                # Check for invalid labels
                if data.y.max() >= out.size(1) or data.y.min() < 0:
                    warnings.warn(f"Invalid labels detected: range {data.y.min()}-{data.y.max()}, expected 0-{out.size(1)-1}")
                    # Skip this batch or clip labels
                    data.y = torch.clamp(data.y, 0, out.size(1) - 1)
                
                loss = self.criterion(out, data.y)
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    warnings.warn(f"Invalid loss in batch {batch_idx}: {loss}")
                    failed_batches += 1
                    continue

                if train:
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Accumulate metrics
                total_loss += loss.item() * data.num_nodes
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.num_nodes
                
                preds_all.append(pred.cpu())
                labels_all.append(data.y.cpu())
                successful_batches += 1
                
            except Exception as e:
                warnings.warn(f"Error in batch {batch_idx}: {e}")
                failed_batches += 1
                continue

        if successful_batches == 0:
            warnings.warn("No successful batches processed!")
            return 0.0, 0.0, [], []
        
        if failed_batches > 0:
            warnings.warn(f"Failed batches: {failed_batches}/{failed_batches + successful_batches}")

        avg_loss = total_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        
        preds_all = torch.cat(preds_all).tolist() if preds_all else []
        labels_all = torch.cat(labels_all).tolist() if labels_all else []
        
        return avg_loss, acc, preds_all, labels_all

    def train_epoch(self, loader):
        loss, acc, _, _ = self._loop(loader, train=True)
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        return loss, acc

    def validate_epoch(self, loader):
        with torch.no_grad():
            loss, acc, preds, labels = self._loop(loader, train=False)
            self.val_losses.append(loss)
            self.val_accs.append(acc)

            # Update scheduler
            self.scheduler.step(acc)
            
            return loss, acc, preds, labels


# ─────────────────────── PyG Data-Loader Factory ─────────────────────────
def create_advanced_data_loaders(dataset,
                                cfg: UserConfig = UserConfig(),
                                split_ratios=(0.8, 0.1, 0.1),
                                shuffle=True, 
                                seed=42):
    """
    Enhanced data loader creation with better error handling and validation.
    """
    print("Building graphs from dataset...")
    builder = GraphBuilder(
        k=cfg.K_NEIGHBORS, 
        max_nodes=cfg.MAX_NODES, 
        min_nodes=cfg.MIN_NODES
    )

    # Build all graphs with progress tracking
    graphs: List[Data] = []
    failed_count = 0
    
    print(f"Processing {len(dataset)} samples...")
    
    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(dataset)} samples")
            
        try:
            samp = dataset[idx]
            img_tensor, target = samp
            
            graph_data = builder.build({
                'boxes': target['boxes'].numpy(),
                'labels': target['labels'].numpy()
            })
            
            if graph_data is not None:
                graphs.append(graph_data)
            else:
                failed_count += 1
                
        except Exception as e:
            warnings.warn(f"Failed to build graph for sample {idx}: {e}")
            failed_count += 1
            continue

    print(f"Successfully built {len(graphs)} graphs, failed: {failed_count}")
    
    if len(graphs) == 0:
        raise ValueError("No valid graphs were created!")
    
    # Filter out graphs that are too small or too large
    original_count = len(graphs)
    graphs = [g for g in graphs if cfg.MIN_NODES <= g.num_nodes <= cfg.MAX_NODES]
    filtered_count = original_count - len(graphs)
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} graphs due to size constraints")

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(graphs)

    # Split dataset
    n_total = len(graphs)
    n_train = int(split_ratios[0] * n_total)
    n_val = int(split_ratios[1] * n_total)

    train_set = graphs[:n_train]
    val_set = graphs[n_train:n_train + n_val]
    test_set = graphs[n_train + n_val:]

    print(f"Split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

    # Create loaders with better configuration
    train_loader = GeoLoader(
        train_set, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = GeoLoader(
        val_set, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = GeoLoader(
        test_set, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    num_features = cfg.HIDDEN_DIMS
    return train_loader, val_loader, test_loader, num_features


# ───────────────────────────── Plotting utils ────────────────────────────
def plot_advanced_results(tr_losses, val_losses, tr_accs, val_accs, 
                          y_true, y_pred, class_names: List[str]):
    """Enhanced plotting with better error handling"""
    try:
        if not tr_losses or not val_losses:
            print("No training data to plot")
            return
            
        epochs = list(range(1, len(tr_losses) + 1))
        
        # Training curves
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].plot(epochs, tr_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, tr_accs, 'b-', label='Train Acc', linewidth=2, alpha=0.8)
        axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2, alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print(" Saved curves to training_curves.png")
        plt.close()

        # Confusion matrix
        if y_true and y_pred and len(y_true) > 0 and len(y_pred) > 0:
            try:
                # Ensure we have valid class indices
                unique_true = set(y_true)
                unique_pred = set(y_pred)
                all_classes = unique_true.union(unique_pred)
                
                if len(all_classes) > 1:
                    cm = confusion_matrix(y_true, y_pred, labels=sorted(all_classes))
                    
                    # Create display labels
                    display_labels = [class_names[i] if i < len(class_names) else f"Class_{i}" 
                                    for i in sorted(all_classes)]
                    
                    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')
                    plt.title('Confusion Matrix')
                    plt.tight_layout()
                    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
                    print(" Saved confusion matrix to confusion_matrix.png")
                    plt.close()
                else:
                    print("Cannot create confusion matrix: only one class present")
                    
            except Exception as e:
                print(f"Could not create confusion matrix: {e}")
                
    except Exception as e:
        print(f"Error in plotting: {e}")