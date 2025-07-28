# """
# End-to-end DocLayNet GCN training with GPU support.

# Usage:
#     python complete_training.py --dataset_size small   # Train on 1% subset
#     python complete_training.py --dataset_size full    # Train on full dataset
#     python complete_training.py --dataset_size small --quick_test --plots

# Key improvements:
# - Robust GPU detection and usage
# - Better error handling and logging
# - Memory management
# - Progress tracking
# - Model checkpointing
# - Fixed import structure
# """

# import os
# import argparse
# import warnings
# import torch
# import numpy as np
# import time
# from pathlib import Path
# import psutil
# import gc

# # Suppress warnings for cleaner output
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)

# # Import dataset utilities - use relative imports
# try:
#     from fixed_data_loader import create_torchvision_loaders, DocLayNetDataset
# except ImportError:
#     try:
#         from doclaynet_data_loader import create_torchvision_loaders, DocLayNetDataset
#     except ImportError as e:
#         raise ImportError(
#             "Could not import data loader. Make sure fixed_data_loader.py or doclaynet_data_loader.py is in the same directory."
#         ) from e

# # Import GCN model and training utilities
# try:
#     from fixed_gcn_model import (
#         TransformerEnhancedGCN,
#         AdvancedTrainer,
#         create_advanced_data_loaders,
#         plot_advanced_results,
#         UserConfig
#     )
# except ImportError:
#     try:
#         from advanced_gcn_model import (
#             TransformerEnhancedGCN,
#             AdvancedTrainer,
#             create_advanced_data_loaders,
#             plot_advanced_results,
#             UserConfig
#         )
#     except ImportError as e:
#         raise ImportError(
#             "Could not import GCN modules. Make sure fixed_gcn_model.py or advanced_gcn_model.py is in the same directory."
#         ) from e


# def print_system_info():
#     """Print system and GPU information"""
#     print("=" * 80)
#     print("SYSTEM INFORMATION")
#     print("=" * 80)
    
#     # CPU info
#     print(f"CPU: {psutil.cpu_count()} cores")
#     print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
#     # GPU info
#     if torch.cuda.is_available():
#         print(f"CUDA Version: {torch.version.cuda}")
#         print(f"PyTorch CUDA: {torch.version.cuda}")
        
#         for i in range(torch.cuda.device_count()):
#             props = torch.cuda.get_device_properties(i)
#             print(f"GPU {i}: {props.name}")
#             print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
#             print(f"  Compute Capability: {props.major}.{props.minor}")
#     else:
#         print("CUDA: Not available")
    
#     print("=" * 80)


# def verify_dataset_structure(dataset_size: str):
#     """Verify that required dataset directories exist"""
#     subset_name = 'doclaynet_1percent' if dataset_size == 'small' else 'doclaynet'
    
#     required_paths = [
#         Path(subset_name) / 'COCO',
#         Path(subset_name) / 'PNG'
#     ]
    
#     missing_paths = [p for p in required_paths if not p.exists()]
    
#     if missing_paths:
#         print("ERROR: Missing required dataset directories:")
#         for path in missing_paths:
#             print(f"  - {path}")
#         print(f"\nExpected structure for {dataset_size} dataset:")
#         print(f"  {subset_name}/")
#         print(f"    ├── COCO/")
#         print(f"    │   ├── train.json")
#         print(f"    │   ├── val.json")
#         print(f"    │   └── test.json")
#         print(f"    └── PNG/")
#         print(f"        └── *.png files")
#         raise FileNotFoundError("Dataset structure verification failed")
    
#     print(f"✓ Dataset structure verified for {dataset_size} dataset")


# def setup_device():
#     """Setup and return the best available device"""
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         # Clear cache and set memory fraction
#         torch.cuda.empty_cache()
#         print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        
#         # Print memory info
#         memory_allocated = torch.cuda.memory_allocated() / (1024**3)
#         memory_reserved = torch.cuda.memory_reserved() / (1024**3)
#         memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
#         print(f"  Memory - Allocated: {memory_allocated:.2f}GB, "
#               f"Reserved: {memory_reserved:.2f}GB, "
#               f"Total: {memory_total:.2f}GB")
#     else:
#         device = torch.device('cpu')
#         print("⚠ Using CPU (CUDA not available)")
    
#     return device


# def run_quick_test(dataset_size: str):
#     """Run a quick test to verify data loading works"""
#     print("\nRunning quick test...")
    
#     try:
#         train_loader, val_loader = create_torchvision_loaders(
#             root='.', 
#             dataset_size=dataset_size, 
#             batch_size=2, 
#             num_workers=0  # Use 0 for debugging
#         )
        
#         # Test one batch
#         images, targets = next(iter(train_loader))
#         print(f"✓ Successfully loaded batch with {len(images)} samples")
        
#         for i, (img, target) in enumerate(zip(images, targets)):
#             print(f"  Sample {i}: Image {img.shape}, "
#                   f"Boxes {target['boxes'].shape}, "
#                   f"Labels {len(target['labels'])}")
        
#         return True
        
#     except Exception as e:
#         print(f"✗ Quick test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return False


# def train_model(dataset_size: str, device: torch.device, config: UserConfig):
#     """Main training function"""
#     print(f"\nStarting training on {dataset_size} dataset...")
    
#     # Create base dataset for graph conversion
#     print("Loading base dataset...")
#     base_dataset = DocLayNetDataset(
#         root='.', 
#         split='train', 
#         dataset_size=dataset_size
#     )
    
#     # Get dataset info
#     num_classes = base_dataset.get_num_classes()
#     class_names = base_dataset.get_class_names()
#     stats = base_dataset.get_stats()
    
#     print(f"Dataset info:")
#     print(f"  Classes: {num_classes}")
#     print(f"  Class names: {class_names}")
#     print(f"  Total images: {stats['num_images']}")
#     print(f"  Total annotations: {stats['num_annotations']}")
    
#     # Create graph data loaders
#     print("\nCreating graph data loaders...")
#     train_loader, val_loader, test_loader, feat_dim = create_advanced_data_loaders(
#         base_dataset, 
#         cfg=config,
#         split_ratios=(0.8, 0.1, 0.1),
#         shuffle=True,
#         seed=42
#     )
    
#     if len(train_loader) == 0:
#         raise ValueError("No training data available!")
    
#     print(f"Data loaders created:")
#     print(f"  Train batches: {len(train_loader)}")
#     print(f"  Val batches: {len(val_loader)}")
#     print(f"  Test batches: {len(test_loader)}")
    
#     # Create model
#     print(f"\nCreating model with {feat_dim} features, {num_classes} classes...")
#     model = TransformerEnhancedGCN(
#         num_features=feat_dim,
#         num_classes=num_classes,
#         config=config
#     )
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
#     # Create trainer
#     trainer = AdvancedTrainer(model, device, config)
    
#     # Training loop with early stopping
#     print(f"\nStarting training for up to {config.EPOCHS} epochs...")
#     print("=" * 80)
    
#     best_val_acc = 0.0
#     patience_counter = 0
#     start_time = time.time()
    
#     for epoch in range(config.EPOCHS):
#         epoch_start = time.time()
        
#         # Training step
#         train_loss, train_acc = trainer.train_epoch(train_loader)
        
#         # Validation step
#         val_loss, val_acc, _, _ = trainer.validate_epoch(val_loader)
        
#         epoch_time = time.time() - epoch_start
        
#         # Print progress
#         print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
#               f"Train: {train_loss:.4f}/{train_acc:.4f} | "
#               f"Val: {val_loss:.4f}/{val_acc:.4f} | "
#               f"Time: {epoch_time:.1f}s")
        
#         # Check for improvement
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             patience_counter = 0
            
#             # Save best model
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': trainer.optimizer.state_dict(),
#                 'val_acc': val_acc,
#                 'config': config.__dict__
#             }, 'best_model.pth')
#             print(f"  → New best model saved! Val acc: {val_acc:.4f}")
            
#         else:
#             patience_counter += 1
            
#         # Early stopping check
#         if patience_counter >= config.EARLY_STOP_PATIENCE:
#             print(f"\nEarly stopping triggered after {epoch+1} epochs")
#             print(f"Best validation accuracy: {best_val_acc:.4f}")
#             break
        
#         # Memory cleanup
#         if device.type == 'cuda':
#             torch.cuda.empty_cache()
#             gc.collect()
    
#     total_time = time.time() - start_time
#     print("=" * 80)
#     print(f"Training completed in {total_time:.1f}s")
#     print(f"Best validation accuracy: {best_val_acc:.4f}")
    
#     return trainer, test_loader, class_names, best_val_acc


# def evaluate_model(trainer, test_loader, class_names, device):
#     """Evaluate the best model on test set"""
#     print("\nEvaluating on test set...")
    
#     # Load best model
#     if os.path.exists('best_model.pth'):
#         checkpoint = torch.load('best_model.pth', map_location=device)
#         trainer.model.load_state_dict(checkpoint['model_state_dict'])
#         print("✓ Loaded best model checkpoint")
#     else:
#         print("⚠ No checkpoint found, using current model")
    
#     # Test evaluation
#     test_loss, test_acc, y_pred, y_true = trainer.validate_epoch(test_loader)
    
#     print(f"Test Results:")
#     print(f"  Loss: {test_loss:.4f}")
#     print(f"  Accuracy: {test_acc:.4f}")
    
#     # Per-class accuracy if we have predictions
#     if y_pred and y_true and len(set(y_true)) > 1:
#         try:
#             from sklearn.metrics import classification_report
#             report = classification_report(
#                 y_true, y_pred, 
#                 target_names=class_names,
#                 zero_division=0
#             )
#             print("\nClassification Report:")
#             print(report)
#         except Exception as e:
#             print(f"Could not generate classification report: {e}")
    
#     return test_acc, y_pred, y_true


# def main(args):
#     """Main training pipeline"""
#     print_system_info()
    
#     # Verify dataset structure
#     verify_dataset_structure(args.dataset_size)
    
#     # Setup device
#     device = setup_device()
    
#     # Quick test if requested
#     if args.quick_test:
#         if not run_quick_test(args.dataset_size):
#             print("Quick test failed, exiting...")
#             return 1
#         print("Quick test passed, continuing with training...\n")
    
#     # Configuration
#     config = UserConfig()
    
#     # Adjust batch size based on dataset size and GPU memory
#     if device.type == 'cuda':
#         # Increase batch size for GPU
#         if args.dataset_size == 'small':
#             config.BATCH_SIZE = 16
#         else:
#             config.BATCH_SIZE = 8
#     else:
#         # Conservative batch size for CPU
#         config.BATCH_SIZE = 4
    
#     print(f"Configuration:")
#     print(f"  Batch size: {config.BATCH_SIZE}")
#     print(f"  Learning rate: {config.LEARNING_RATE}")
#     print(f"  Hidden dims: {config.HIDDEN_DIMS}")
#     print(f"  Max epochs: {config.EPOCHS}")
#     print(f"  Early stopping patience: {config.EARLY_STOP_PATIENCE}")
    
#     try:
#         # Train model
#         trainer, test_loader, class_names, best_val_acc = train_model(args.dataset_size, device, config)
        
#         # Evaluate model
#         test_acc, y_pred, y_true = evaluate_model(trainer, test_loader, class_names, device)
        
#         # Generate plots if requested
#         if args.plots:
#             print("\nGenerating plots...")
#             plot_advanced_results(
#                 trainer.train_losses, trainer.val_losses,
#                 trainer.train_accs, trainer.val_accs,
#                 y_true, y_pred, class_names
#             )
        
#         # Final summary
#         print("\n" + "=" * 80)
#         print("TRAINING SUMMARY")
#         print("=" * 80)
#         print(f"Dataset: {args.dataset_size}")
#         print(f"Device: {device}")
#         print(f"Best validation accuracy: {best_val_acc:.4f}")
#         print(f"Test accuracy: {test_acc:.4f}")
#         print(f"Model saved as: best_model.pth")
#         if args.plots:
#             print(f"Plots saved as: training_curves.png, confusion_matrix.png")
#         print("=" * 80)
        
#     except Exception as e:
#         print(f"\nError during training: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
    
#     return 0


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Train GCN model on DocLayNet dataset',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
    
#     parser.add_argument(
#         '--dataset_size', 
#         choices=['small', 'full'], 
#         default='small',
#         help='Dataset size: small (1%%) or full (100%%)'
#     )
    
#     parser.add_argument(
#         '--quick_test', 
#         action='store_true',
#         help='Run quick data loading test before training'
#     )
    
#     parser.add_argument(
#         '--plots', 
#         action='store_true',
#         help='Generate training curves and confusion matrix plots'
#     )
    
#     args = parser.parse_args()
    
#     # Set random seeds for reproducibility
#     torch.manual_seed(42)
#     np.random.seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(42)
    
#     exit_code = main(args)
#     exit(exit_code)




"""
End-to-end DocLayNet Advanced GCN training with GPU support.

Usage:
    python complete_training.py --dataset_size small   # Train on 1% subset
    python complete_training.py --dataset_size full    # Train on full dataset
    python complete_training.py --dataset_size small --quick_test --plots

Key improvements:
- Uses TransformerEnhancedGCN (the actual advanced model class)
- Robust GPU detection and usage
- Better error handling and logging
- Memory management optimized for larger models
- Progress tracking
- Model checkpointing
- Fixed import structure for advanced model
"""

import os
import argparse
import warnings
import torch
import numpy as np
import time
from pathlib import Path
import psutil
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import dataset utilities - use relative imports
try:
    from fixed_data_loader import create_torchvision_loaders, DocLayNetDataset
except ImportError:
    try:
        from doclaynet_data_loader import create_torchvision_loaders, DocLayNetDataset
    except ImportError as e:
        raise ImportError(
            "Could not import data loader. Make sure fixed_data_loader.py or doclaynet_data_loader.py is in the same directory."
        ) from e

# Import Advanced GCN model and training utilities - CORRECTED IMPORTS
try:
    from advanced_gcn_model import (
        TransformerEnhancedGCN,      # This is the actual class name in your file
        AdvancedTrainer,
        create_advanced_data_loaders,
        plot_advanced_results,
        UserConfig
    )
    print(" Successfully imported from advanced_gcn_model.py")
except ImportError:
    try:
        from fixed_gcn_model import (
            TransformerEnhancedGCN,  # Fallback to fixed model
            AdvancedTrainer,
            create_advanced_data_loaders,
            plot_advanced_results,
            UserConfig
        )
        print("Successfully imported from fixed_gcn_model.py (fallback)")
    except ImportError as e:
        raise ImportError(
            "Could not import Advanced GCN modules. Make sure advanced_gcn_model.py is in the same directory."
        ) from e

def print_system_info():
    """Print system and GPU information with advanced model requirements"""
    print("=" * 80)
    print("TRANSFORMER-ENHANCED GCN MODEL - SYSTEM INFORMATION")
    print("=" * 80)
    
    # CPU info
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU info with advanced model memory requirements
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Advanced model memory recommendations
            gpu_memory = props.total_memory / (1024**3)
            if gpu_memory >= 8:
                print(f"   Suitable for TransformerEnhanced GCN (recommended: 8GB+)")
            elif gpu_memory >= 6:
                print(f"   Marginal for TransformerEnhanced GCN (may need smaller batch sizes)")
            else:
                print(f"   Limited for TransformerEnhanced GCN (consider using standard model)")
    else:
        print("CUDA: Not available")
        print(" TransformerEnhanced GCN will be significantly slower on CPU")
    
    print("=" * 80)

def verify_dataset_structure(dataset_size: str):
    """Verify that required dataset directories exist"""
    subset_name = 'doclaynet_1percent' if dataset_size == 'small' else 'doclaynet'
    
    required_paths = [
        Path(subset_name) / 'COCO',
        Path(subset_name) / 'PNG'
    ]
    
    missing_paths = [p for p in required_paths if not p.exists()]
    
    if missing_paths:
        print("ERROR: Missing required dataset directories:")
        for path in missing_paths:
            print(f"  - {path}")
        print(f"\nExpected structure for {dataset_size} dataset:")
        # print(f"  {subset_name}/")
        # print(f"    ├── COCO/")
        # print(f"    │   ├── train.json")
        # print(f"    │   ├── val.json")
        # print(f"    │   └── test.json")
        # print(f"    └── PNG/")
        # print(f"        └── *.png files")
        raise FileNotFoundError("Dataset structure verification failed")
    
    print(f" Dataset structure verified for {dataset_size} dataset")

def setup_device():
    """Setup and return the best available device with advanced model considerations"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Clear cache and set memory fraction for larger model
        torch.cuda.empty_cache()
        print(f" Using GPU for TransformerEnhanced GCN: {torch.cuda.get_device_name()}")
        
        # Print memory info
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"  Memory - Allocated: {memory_allocated:.2f}GB, "
              f"Reserved: {memory_reserved:.2f}GB, "
              f"Total: {memory_total:.2f}GB")
        
        # Warn if memory might be insufficient
        if memory_total < 6:
            print("   Warning: TransformerEnhanced GCN may require more GPU memory")
            
    else:
        device = torch.device('cpu')
        print("Using CPU for TransformerEnhanced GCN (training will be much slower)")
    
    return device

def run_quick_test(dataset_size: str):
    """Run a quick test to verify data loading works"""
    print("\nRunning quick test...")
    
    try:
        train_loader, val_loader = create_torchvision_loaders(
            root='.', 
            dataset_size=dataset_size, 
            batch_size=2, 
            num_workers=0  # Use 0 for debugging
        )
        
        # Test one batch
        images, targets = next(iter(train_loader))
        print(f" Successfully loaded batch with {len(images)} samples")
        
        for i, (img, target) in enumerate(zip(images, targets)):
            print(f"  Sample {i}: Image {img.shape}, "
                  f"Boxes {target['boxes'].shape}, "
                  f"Labels {len(target['labels'])}")
        
        return True
        
    except Exception as e:
        print(f" Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_model(dataset_size: str, device: torch.device, config: UserConfig):
    """Main training function with TransformerEnhanced GCN"""
    print(f"\nStarting TransformerEnhanced GCN training on {dataset_size} dataset...")
    
    # Create base dataset for graph conversion
    print("Loading base dataset...")
    base_dataset = DocLayNetDataset(
        root='.', 
        split='train', 
        dataset_size=dataset_size
    )
    
    # Get dataset info
    num_classes = base_dataset.get_num_classes()
    class_names = base_dataset.get_class_names()
    stats = base_dataset.get_stats()
    
    print(f"Dataset info:")
    print(f"  Classes: {num_classes}")
    print(f"  Class names: {class_names}")
    print(f"  Total images: {stats['num_images']}")
    print(f"  Total annotations: {stats['num_annotations']}")
    
    # Create graph data loaders
    print("\nCreating graph data loaders for TransformerEnhanced GCN...")
    train_loader, val_loader, test_loader, feat_dim = create_advanced_data_loaders(
        base_dataset, 
        cfg=config,
        split_ratios=(0.8, 0.1, 0.1),
        shuffle=True,
        seed=42
    )
    
    if len(train_loader) == 0:
        raise ValueError("No training data available!")
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Feature dimensions: {feat_dim}")
    
    # Create TransformerEnhanced GCN model - CORRECTED MODEL INSTANTIATION
    print(f"\nCreating TransformerEnhanced GCN model with {feat_dim} features, {num_classes} classes...")
    model = TransformerEnhancedGCN(
        num_features=feat_dim,    # This matches the constructor in your advanced_gcn_model.py
        num_classes=num_classes,
        config=config             # This passes the UserConfig object
    )
    
    # Move model to device
    model = model.to(device)
    
    # Count parameters (should be significantly higher than standard model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransformerEnhanced GCN parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Model architecture: 2-layer GCN → {config.TRANSFORMER_LAYERS}-layer Transformer → Classifier")
    print(f"  Hidden dimensions: {config.HIDDEN_DIMS}")
    print(f"  Transformer heads: {config.NUM_HEADS}")
    
    # Create trainer with advanced model considerations
    trainer = AdvancedTrainer(model, device, config)
    
    # Training loop with early stopping
    print(f"\nStarting TransformerEnhanced GCN training for up to {config.EPOCHS} epochs...")
    print("Note: Training will be slower due to Transformer components")
    print("=" * 80)
    
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        
        # Training step
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validation step
        val_loss, val_acc, _, _ = trainer.validate_epoch(val_loader)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress with timing info
        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config.__dict__,
                'model_type': 'TransformerEnhancedGCN',
                'model_params': total_params
            }, 'best_transformer_gcn_model.pth')
            print(f"   New best TransformerEnhanced GCN model saved! Val acc: {val_acc:.4f}")
            
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
        
        # Enhanced memory cleanup for larger model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"TransformerEnhanced GCN training completed in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return trainer, test_loader, class_names, best_val_acc

def evaluate_model(trainer, test_loader, class_names, device):
    """Evaluate the best TransformerEnhanced GCN model on test set"""
    print("\nEvaluating TransformerEnhanced GCN on test set...")
    
    # Load best model
    checkpoint_file = 'best_transformer_gcn_model.pth'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best TransformerEnhanced GCN model checkpoint")
        print(f"  Model type: {checkpoint.get('model_type', 'Unknown')}")
        print(f"  Parameters: {checkpoint.get('model_params', 'Unknown'):,}")
    else:
        print(" No checkpoint found, using current model")
    
    # Test evaluation
    test_loss, test_acc, y_pred, y_true = trainer.validate_epoch(test_loader)
    
    print(f"TransformerEnhanced GCN Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Per-class accuracy if we have predictions
    if y_pred and y_true and len(set(y_true)) > 1:
        try:
            from sklearn.metrics import classification_report
            report = classification_report(
                y_true, y_pred, 
                target_names=class_names,
                zero_division=0
            )
            print("\nTransformerEnhanced GCN Classification Report:")
            print(report)
        except Exception as e:
            print(f"Could not generate classification report: {e}")
    
    return test_acc, y_pred, y_true

def main(args):
    """Main training pipeline for TransformerEnhanced GCN"""
    print("TRANSFORMER-ENHANCED GCN MODEL TRAINING")
    print_system_info()
    
    # Verify dataset structure
    verify_dataset_structure(args.dataset_size)
    
    # Setup device
    device = setup_device()
    
    # Quick test if requested
    if args.quick_test:
        if not run_quick_test(args.dataset_size):
            print("Quick test failed, exiting...")
            return 1
        print("Quick test passed, continuing with TransformerEnhanced GCN training...\n")
    
    # Configuration optimized for TransformerEnhanced GCN
    config = UserConfig()
    
    # Advanced model specific adjustments
    config.HIDDEN_DIMS = 128          # As defined in your advanced_gcn_model.py
    config.NUM_HEADS = 4              # Transformer attention heads
    config.TRANSFORMER_LAYERS = 2     # Transformer encoder layers
    config.DROPOUT = 0.1              # Dropout rate
    config.LEARNING_RATE = 3e-4       # Learning rate from your config
    config.WEIGHT_DECAY = 1e-4        # Weight decay
    config.K_NEIGHBORS = 10           # KNN edges per node
    config.MAX_NODES = 300            # Max nodes per graph
    
    # Adjust batch size based on dataset size and GPU memory (smaller for advanced model)
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory >= 8:
            config.BATCH_SIZE = 8 if args.dataset_size == 'small' else 4
        else:
            config.BATCH_SIZE = 4 if args.dataset_size == 'small' else 2
    else:
        # Very conservative batch size for CPU
        config.BATCH_SIZE = 2
    
    print(f"TransformerEnhanced GCN Configuration:")
    print(f"  Batch size: {config.BATCH_SIZE} (optimized for model complexity)")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Hidden dims: {config.HIDDEN_DIMS}")
    print(f"  Transformer heads: {config.NUM_HEADS}")
    print(f"  Transformer layers: {config.TRANSFORMER_LAYERS}")
    print(f"  Dropout: {config.DROPOUT}")
    print(f"  K-neighbors: {config.K_NEIGHBORS}")
    print(f"  Max nodes per graph: {config.MAX_NODES}")
    print(f"  Max epochs: {config.EPOCHS}")
    print(f"  Early stopping patience: {config.EARLY_STOP_PATIENCE}")
    
    try:
        # Train TransformerEnhanced GCN model
        trainer, test_loader, class_names, best_val_acc = train_model(args.dataset_size, device, config)
        
        # Evaluate model
        test_acc, y_pred, y_true = evaluate_model(trainer, test_loader, class_names, device)
        
        # Generate plots if requested
        if args.plots:
            print("\nGenerating TransformerEnhanced GCN plots...")
            try:
                plot_advanced_results(
                    trainer.train_losses, trainer.val_losses,
                    trainer.train_accs, trainer.val_accs,
                    y_true, y_pred, class_names
                )
            except Exception as e:
                print(f"Could not generate plots: {e}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("TRANSFORMER-ENHANCED GCN TRAINING SUMMARY")
        print("=" * 80)
        print(f"Model: TransformerEnhanced GCN (2-layer GCN + {config.TRANSFORMER_LAYERS}-layer Transformer)")
        print(f"Dataset: {args.dataset_size}")
        print(f"Device: {device}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Model saved as: best_transformer_gcn_model.pth")
        if args.plots:
            print(f"Plots saved as: training_curves.png, confusion_matrix.png")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during TransformerEnhanced GCN training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train TransformerEnhanced GCN model on DocLayNet dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset_size', 
        choices=['small', 'full'], 
        default='small',
        help='Dataset size: small (1%%) or full (100%%)'
    )
    
    parser.add_argument(
        '--quick_test', 
        action='store_true',
        help='Run quick data loading test before training'
    )
    
    parser.add_argument(
        '--plots', 
        action='store_true',
        help='Generate training curves and confusion matrix plots'
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    exit_code = main(args)
    exit(exit_code)
