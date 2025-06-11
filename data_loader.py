import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image

class PVDNDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None, config=None):
        self.data_path = data_path
        self.transform = transform
        self.config = config
        
        annotation_file = os.path.join(data_path, f"{split}_annotations.json")
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_ids = list(self.annotations.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        annotation = self.annotations[image_id]
        
        image_path = os.path.join(self.data_path, 'images', f"{image_id}.jpg")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, tuple(self.config['model']['input_size']))
        
        if self.transform:
            image = self.transform(image)
        
        image = torch.FloatTensor(image).unsqueeze(0) / 255.0
        
        bbox_data = self._parse_bboxes(annotation)
        mask = self._create_mask(bbox_data, self.config['model']['input_size'])
        
        return {
            'image': image,
            'mask': torch.FloatTensor(mask),
            'bboxes': torch.FloatTensor(bbox_data),
            'image_id': image_id
        }
    
    def _parse_bboxes(self, annotation):
        bboxes = []
        for vehicle in annotation.get('vehicles', []):
            if vehicle.get('direct', False):
                bbox = vehicle.get('bbox', [0, 0, 0, 0])
                bboxes.append(bbox)
        return np.array(bboxes) if bboxes else np.zeros((0, 4))
    
    def _create_mask(self, bboxes, size):
        mask = np.zeros(size, dtype=np.float32)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            mask[y1:y2, x1:x2] = 1.0
        return mask

def create_dataloaders(config):
    dataset_path = config['data']['dataset_path']
    batch_size = config['training']['batch_size']
    
    from .data_preprocessor import get_transforms
    train_transform = get_transforms(training=True) if config['data']['augmentation'] else None
    val_transform = get_transforms(training=False)
    
    train_dataset = PVDNDataset(dataset_path, 'train', train_transform, config)
    val_dataset = PVDNDataset(dataset_path, 'val', val_transform, config)
    test_dataset = PVDNDataset(dataset_path, 'test', val_transform, config)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader