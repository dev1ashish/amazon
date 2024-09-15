import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import re
from constants import entity_unit_map, allowed_units
from transformers import T5Tokenizer



class ProductDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.all_units = list(allowed_units)
        self.unit_to_idx = {unit: idx for idx, unit in enumerate(self.all_units)}

    def __len__(self):
        return len(self.data)
    
    def parse_entity_value(self, entity_value, entity_name):
        if pd.isna(entity_value) or entity_value == "":
            return 0.0, ""
        match = re.match(r"([-+]?\d*\.?\d+)\s*(.+)", str(entity_value))
        if match:
            value, unit = match.group(1), match.group(2)
            if unit in entity_unit_map.get(entity_name, set()):
                return float(value), unit
        return 0.0, ""

    def normalize_value(self, value, entity_name):
        if entity_name in ['width', 'depth', 'height', 'item_weight', 'maximum_weight_recommendation']:
            return value / 1000
        elif entity_name == 'voltage':
            return value / 1000
        elif entity_name == 'wattage':
            return value / 10000
        elif entity_name == 'item_volume':
            return value / 1000
        return value

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_input = f"Extract {row['entity_name']} from {row['group_id']}: {row.get('product_description', '')}"

        encoded_input = self.tokenizer(
            text_input,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        image_path = os.path.join(self.img_dir, row['image_link'].split('/')[-1])
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                image = torch.zeros((3, 299, 299))
        else:
            image = torch.zeros((3, 299, 299))

        value, unit = self.parse_entity_value(row.get('entity_value', ''), row['entity_name'])
        normalized_value = self.normalize_value(value, row['entity_name'])
        
        unit_index = self.unit_to_idx[unit] if unit else -1
        
        allowed_units = entity_unit_map.get(row['entity_name'], set())
        allowed_units_tensor = torch.tensor([1 if u in allowed_units else 0 for u in self.all_units])

        return {
            'input_ids': encoded_input['input_ids'].squeeze(0),
            'attention_mask': encoded_input['attention_mask'].squeeze(0),
            'image': image,
            'value': torch.tensor(normalized_value, dtype=torch.float),
            'unit': torch.tensor(unit_index, dtype=torch.long),
            'allowed_units': allowed_units_tensor,
            'entity_name': row['entity_name'],
            'index': idx
        }

def load_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")

def prepare_dataset(csv_file, img_dir, tokenizer):
    return ProductDataset(csv_file, img_dir, tokenizer)