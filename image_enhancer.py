import cv2
import numpy as np
import torch
import torch.nn.functional as F

class BioinspiredEnhancer:
    def __init__(self, config):
        self.config = config['preprocessing']
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.get('clahe_clip_limit', 2.0),
            tileGridSize=(8, 8)
        )
    
    def enhance(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        enhanced = self.clahe.apply(image)
        enhanced = self.gamma_correction(enhanced)
        enhanced = self.adaptive_contrast(enhanced)
        enhanced = self.reduce_noise(enhanced)
        
        return enhanced
    
    def gamma_correction(self, image):
        gamma = self.config.get('gamma_correction', 2.2)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def adaptive_contrast(self, image):
        mean_intensity = np.mean(image)
        if mean_intensity < 50:
            alpha = 1.5
        elif mean_intensity < 100:
            alpha = 1.2
        else:
            alpha = 1.0
        
        return np.clip(image * alpha, 0, 255).astype(np.uint8)
    
    def reduce_noise(self, image):
        if self.config.get('noise_reduction', True):
            return cv2.bilateralFilter(image, 9, 75, 75)
        return image

class NightImageProcessor:
    def __init__(self, config):
        self.enhancer = BioinspiredEnhancer(config)
        self.config = config
    
    def process_batch(self, batch_images):
        enhanced_batch = []
        for image in batch_images:
            if torch.is_tensor(image):
                image = image.cpu().numpy()
                image = (image * 255).astype(np.uint8)
            
            enhanced = self.enhancer.enhance(image)
            enhanced_batch.append(enhanced)
        
        return np.array(enhanced_batch)
    
    def process_single(self, image):
        if torch.is_tensor(image):
            image = image.cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        return self.enhancer.enhance(image)

def enhance_low_light_image(image, config):
    processor = NightImageProcessor(config)
    return processor.process_single(image)