import cv2
import numpy as np
from torchvision import transforms
import torch

class NightImageAugmentation:
    def __init__(self, training=True):
        self.training = training
    
    def __call__(self, image):
        if self.training:
            image = self.random_brightness(image)
            image = self.random_contrast(image)
            image = self.random_noise(image)
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
        return image
    
    def random_brightness(self, image, factor=0.2):
        alpha = 1.0 + np.random.uniform(-factor, factor)
        return np.clip(image * alpha, 0, 255).astype(np.uint8)
    
    def random_contrast(self, image, factor=0.3):
        mean = image.mean()
        alpha = 1.0 + np.random.uniform(-factor, factor)
        return np.clip((image - mean) * alpha + mean, 0, 255).astype(np.uint8)
    
    def random_noise(self, image, noise_level=10):
        noise = np.random.normal(0, noise_level, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

class ImageNormalizer:
    def __init__(self):
        self.mean = 0.45
        self.std = 0.225
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.FloatTensor(image)
        return (image - self.mean) / self.std

def apply_clahe(image, clip_limit=2.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image)

def gamma_correction(image, gamma=2.2):
    gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, gamma_table)

def reduce_noise(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def preprocess_night_image(image, config):
    preprocessing_config = config['preprocessing']
    
    if preprocessing_config['clahe_clip_limit']:
        image = apply_clahe(image, preprocessing_config['clahe_clip_limit'])
    
    if preprocessing_config['gamma_correction']:
        image = gamma_correction(image, preprocessing_config['gamma_correction'])
    
    if preprocessing_config['noise_reduction']:
        image = reduce_noise(image)
    
    return image

def get_transforms(training=True):
    transform_list = []
    
    if training:
        transform_list.append(NightImageAugmentation(training=True))
    
    transform_list.extend([
        ImageNormalizer()
    ])
    
    return transforms.Compose(transform_list)