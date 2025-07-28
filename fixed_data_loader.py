"""
Flexible DocLayNet loader (torch-vision / PyG ready).

* dataset_size = 'small' → expects 1% subset folder  doclaynet_1percent/
* dataset_size = 'full'  → expects full dataset      doclaynet/

Key improvements:
- Better error handling and validation
- Robust box coordinate processing
- Memory-efficient loading
- GPU tensor handling
"""

import os, json, torch, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from typing import Dict, List, Tuple, Optional
import warnings


class DocLayNetDataset(Dataset):
    """
    DocLayNet dataset loader that returns:
    - img: (C,H,W) float32 tensor in [0,1]
    - target: dict with boxes, labels, image_id, file_name
    """

    def __init__(self, root='.', split='train', 
                 dataset_size='full', transforms=None):
        """
        Args:
            root: Root directory containing dataset
            split: 'train', 'val', or 'test'
            dataset_size: 'small' (1%) or 'full' (100%)
            transforms: Optional transform function
        """
        subset = 'doclaynet_1percent' if dataset_size == 'small' else 'doclaynet'
        self.dataset_size = dataset_size
        self.split = split
        
        # Paths
        coco_file = os.path.join(root, subset, 'COCO', f'{split}.json')
        self.png_dir = os.path.join(root, subset, 'PNG')

        # Validation
        if not os.path.exists(coco_file):
            raise FileNotFoundError(f'COCO annotation file not found: {coco_file}')
        if not os.path.isdir(self.png_dir):
            raise FileNotFoundError(f'PNG directory not found: {self.png_dir}')

        # Load COCO annotations
        print(f"Loading {split} annotations from {coco_file}...")
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # Create lookup dictionaries
        self.id_to_image = {img['id']: img for img in coco_data['images']}
        self.id_to_annotations = {}
        
        # Group annotations by image_id
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id_to_annotations:
                self.id_to_annotations[img_id] = []
            self.id_to_annotations[img_id].append(ann)

        # Filter to only images that have annotations
        self.image_ids = [img_id for img_id in self.id_to_image.keys() 
                         if img_id in self.id_to_annotations]
        
        # Category mapping
        self.category_id_to_name = {cat['id']: cat['name'] 
                                   for cat in coco_data['categories']}
        self.category_names = list(self.category_id_to_name.values())
        
        # Transform function
        self.transforms = transforms

        print(f'[{split}] {subset}: {len(self.image_ids)} images | '
              f'{len(coco_data["annotations"])} annotations | '
              f'{len(self.category_names)} classes')
        print(f'Classes: {self.category_names}')

    def _load_image_as_tensor(self, file_path: str) -> torch.Tensor:
        """Load PNG image and convert to tensor"""
        try:
            # Load and convert to RGB
            with Image.open(file_path) as img:
                img_rgb = img.convert('RGB')
                # Convert to tensor [0,1] range
                tensor = F.pil_to_tensor(img_rgb).float() / 255.0
                return tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load image {file_path}: {e}")

    def _validate_and_fix_boxes(self, boxes: List[List[float]], 
                               img_width: int, img_height: int) -> List[List[float]]:
        """Validate and fix bounding boxes"""
        fixed_boxes = []
        
        for box in boxes:
            x, y, w, h = box
            
            # Ensure positive width/height
            w = max(w, 1.0)
            h = max(h, 1.0)
            
            # Clamp to image bounds
            x = max(0, min(x, img_width - w))
            y = max(0, min(y, img_height - h))
            
            # Convert xywh to xyxy
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            # Final validation
            x2 = min(x2, img_width)
            y2 = min(y2, img_height)
            
            if x2 > x1 and y2 > y1:  # Valid box
                fixed_boxes.append([x1, y1, x2, y2])
            else:
                warnings.warn(f"Dropping invalid box: {box}")
                
        return fixed_boxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get item by index"""
        if idx >= len(self.image_ids):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_ids)}")
            
        image_id = self.image_ids[idx]
        image_info = self.id_to_image[image_id]
        
        # Load image
        img_path = os.path.join(self.png_dir, image_info['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        image_tensor = self._load_image_as_tensor(img_path)
        
        # Get image dimensions
        img_height, img_width = image_info.get('height', 1025), image_info.get('width', 1025)
        
        # Process annotations
        annotations = self.id_to_annotations[image_id]
        raw_boxes = [ann['bbox'] for ann in annotations]  # xywh format
        labels = [ann['category_id'] for ann in annotations]
        
        # Validate and fix boxes
        fixed_boxes = self._validate_and_fix_boxes(raw_boxes, img_width, img_height)
        
        # Handle case where all boxes were invalid
        if not fixed_boxes:
            warnings.warn(f"No valid boxes for image {image_id}, creating dummy box")
            fixed_boxes = [[0, 0, 10, 10]]  # Dummy box
            labels = [1]  # Dummy label

        # Keep only labels for valid boxes
        if len(fixed_boxes) != len(labels):
            labels = labels[:len(fixed_boxes)]

        # Create target dictionary
        target = {
            'boxes': torch.as_tensor(fixed_boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'file_name': image_info['file_name'],
            'orig_size': torch.tensor([img_height, img_width])
        }

        # Apply transforms if provided
        if self.transforms is not None:
            try:
                image_tensor, target = self.transforms(image_tensor, target)
            except Exception as e:
                warnings.warn(f"Transform failed for image {image_id}: {e}")

        return image_tensor, target

    def __len__(self) -> int:
        """Dataset length"""
        return len(self.image_ids)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.category_names
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.category_names)

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        total_annotations = sum(len(anns) for anns in self.id_to_annotations.values())
        
        # Count annotations per class
        class_counts = {}
        for anns in self.id_to_annotations.values():
            for ann in anns:
                cat_id = ann['category_id']
                cat_name = self.category_id_to_name.get(cat_id, f"Unknown_{cat_id}")
                class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
        
        return {
            'num_images': len(self.image_ids),
            'num_annotations': total_annotations,
            'num_classes': len(self.category_names),
            'class_names': self.category_names,
            'class_distribution': class_counts,
            'avg_annotations_per_image': total_annotations / len(self.image_ids) if self.image_ids else 0
        }


def create_torchvision_loaders(root='.', dataset_size='small', 
                              batch_size=4, num_workers=0, pin_memory=True):
    """
    Create standard torchvision data loaders
    
    Args:
        root: Dataset root directory
        dataset_size: 'small' or 'full'
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Use pinned memory for GPU transfer
    
    Returns:
        train_loader, val_loader
    """
    print(f"Creating torchvision loaders for {dataset_size} dataset...")
    
    # Create datasets
    train_dataset = DocLayNetDataset(root, 'train', dataset_size)
    val_dataset = DocLayNetDataset(root, 'val', dataset_size)
    
    def collate_fn(batch):
        """Custom collate function for variable-size targets"""
        return tuple(zip(*batch))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )
    
    # Print dataset statistics
    train_stats = train_dataset.get_stats()
    val_stats = val_dataset.get_stats()
    
    print(f"Train: {train_stats['num_images']} images, {train_stats['num_annotations']} annotations")
    print(f"Val: {val_stats['num_images']} images, {val_stats['num_annotations']} annotations")
    print(f"Classes ({train_stats['num_classes']}): {train_stats['class_names']}")
    
    return train_loader, val_loader


# Self-test and debugging
if __name__ == '__main__':
    print("Testing DocLayNet data loader...")
    
    try:
        # Test with small dataset
        train_loader, val_loader = create_torchvision_loaders(
            dataset_size='small', 
            batch_size=2, 
            num_workers=0  # Use 0 for debugging
        )
        
        # Test first batch
        print("\nTesting first batch...")
        images, targets = next(iter(train_loader))
        
        print(f"Batch size: {len(images)}")
        for i, (img, target) in enumerate(zip(images, targets)):
            print(f"  Sample {i}:")
            print(f"    Image shape: {img.shape}")
            print(f"    Boxes shape: {target['boxes'].shape}")
            print(f"    Labels: {target['labels'].unique().tolist()}")
            print(f"    File: {target['file_name']}")
        
        print("\nData loader test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()