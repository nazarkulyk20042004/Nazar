import torch
import torch.nn as nn
import cv2
import numpy as np
from ..models.unet_model import UNetWithAttention
from ..models.loss_functions import CombinedLoss
from ..processing.image_enhancer import NightImageProcessor
from ..processing.cascade_selector import CascadeFeatureSelector
from .postprocessor import PostProcessor

class VehicleDetector:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.model = UNetWithAttention(config).to(self.device)
        self.criterion = CombinedLoss(config)
        self.image_processor = NightImageProcessor(config)
        self.cascade_selector = CascadeFeatureSelector(config)
        self.postprocessor = PostProcessor(config)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['training']['warmup_epochs'],
            T_mult=2
        )
    
    def coarse_detection(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        enhanced_image = self.image_processor.process_single(image)
        regions = self.cascade_selector.apply_sliding_window(enhanced_image)
        
        candidate_regions = []
        for region in regions:
            x, y, w, h = region
            roi = enhanced_image[y:y+h, x:x+w]
            
            mean_intensity = np.mean(roi)
            std_intensity = np.std(roi)
            
            if mean_intensity > 50 and std_intensity > 15:
                candidate_regions.append(region)
        
        return candidate_regions[:50]
    
    def fine_detection(self, image, candidate_regions):
        detections = []
        
        for region in candidate_regions:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            
            roi_resized = cv2.resize(roi, tuple(self.config['model']['input_size']))
            roi_tensor = torch.FloatTensor(roi_resized).unsqueeze(0).unsqueeze(0) / 255.0
            roi_tensor = roi_tensor.to(self.device)
            
            with torch.no_grad():
                predictions = self.model(roi_tensor)
            
            segmentation = predictions['segmentation'].cpu().numpy()[0, 0]
            confidence = predictions['confidence'].cpu().numpy()[0, 0]
            bbox_reg = predictions['bbox_regression'].cpu().numpy()[0]
            
            mean_confidence = np.mean(confidence)
            if mean_confidence > self.config['detection']['confidence_threshold']:
                
                seg_resized = cv2.resize(segmentation, (w, h))
                conf_resized = cv2.resize(confidence, (w, h))
                
                detection = {
                    'bbox': (x, y, x + w, y + h),
                    'confidence': mean_confidence,
                    'segmentation': seg_resized,
                    'region_confidence': conf_resized
                }
                detections.append(detection)
        
        return detections
    
    def detect(self, image):
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        
        if len(image.shape) == 4:
            image = image[0]
        
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image[0]
        
        original_shape = image.shape
        
        enhanced_image = self.image_processor.process_single(image)
        
        candidate_regions = self.coarse_detection(enhanced_image)
        
        detections = self.fine_detection(enhanced_image, candidate_regions)
        
        filtered_detections = self.postprocessor.apply_nms(detections)
        
        return {
            'detections': filtered_detections,
            'enhanced_image': enhanced_image,
            'candidate_regions': candidate_regions
        }
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        images = batch['image'].to(self.device)
        targets = {
            'mask': batch['mask'].to(self.device),
            'bboxes': batch['bboxes'].to(self.device)
        }
        
        predictions = self.model(images)
        losses = self.criterion(predictions, targets)
        
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return losses
    
    def validation_step(self, batch):
        self.model.eval()
        
        with torch.no_grad():
            images = batch['image'].to(self.device)
            targets = {
                'mask': batch['mask'].to(self.device),
                'bboxes': batch['bboxes'].to(self.device)
            }
            
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
        
        return losses, predictions
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def set_model_mode(self, mode):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()