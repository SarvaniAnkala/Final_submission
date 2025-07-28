import os
import json
from pdf2image import convert_from_path
from PIL import Image
from advanced_gcn_model import TransformerEnhancedGCN, UserConfig, GraphBuilder
from fixed_data_loader import DocLayNetDataset
import torch
import numpy as np

class ModelInference:
    def __init__(self, model_path='best_transformer_gcn_model.pth', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.class_names = [
            'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
            'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
        ]
        self.graph_builder = None
        self.load_model(model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = UserConfig()
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        self.model = TransformerEnhancedGCN(
            num_features=self.config.HIDDEN_DIMS,
            num_classes=len(self.class_names),
            config=self.config
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.graph_builder = GraphBuilder(
            k=self.config.K_NEIGHBORS,
            max_nodes=self.config.MAX_NODES
        )

    def predict(self, image):
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Dummy boxes — replace with actual detection later
        box_size = 100
        boxes = [
            [x, y, x + box_size, y + box_size]
            for y in range(0, height - box_size, box_size)
            for x in range(0, width - box_size, box_size)
        ] or [[0, 0, min(100, width), min(100, height)]]

        labels = [1] * len(boxes)
        sample_data = {'boxes': np.array(boxes, dtype=np.float32),
                       'labels': np.array(labels, dtype=np.int64)}

        graph_data = self.graph_builder.build(sample_data)
        graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
        graph_data = graph_data.to(self.device)

        with torch.no_grad():
            outputs = self.model(graph_data)
            predictions = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)

        results = []
        for i in range(len(sample_data['boxes'])):
            box = sample_data['boxes'][i].tolist()
            pred_class_idx = predicted_classes[i].item()
            confidence = predictions[i][pred_class_idx].item()
            results.append({
                'box': box,
                'class': self.class_names[pred_class_idx],
                'confidence': float(confidence)
            })

        return results

# Paths
input_dir = "input"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

model = ModelInference()

for filename in os.listdir(input_dir):
    if filename.endswith(".pdf"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
        try:
            print(f"Processing {filename}")
            images = convert_from_path(input_path, first_page=1, last_page=1)
            result = model.predict(images[0])
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
