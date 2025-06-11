import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean
from .feature_extractor import FeatureExtractor

class CascadeFeatureSelector:
    def __init__(self, config):
        self.config = config['detection']
        self.scales = config['detection']['cascade_scales']
        self.step = config['detection']['cascade_step']
        self.feature_extractor = FeatureExtractor()
        self.selected_regions = []
        self.region_weights = []
    
    def generate_multiscale_regions(self, image_shape):
        height, width = image_shape[:2]
        regions = []
        
        for scale in self.scales:
            for y in range(0, height - scale, self.step):
                for x in range(0, width - scale, self.step):
                    regions.append((x, y, scale, scale))
        
        return regions
    
    def calculate_region_overlap(self, region1, region2):
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2
        
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        overlap_area = x_overlap * y_overlap
        union_area = w1 * h1 + w2 * h2 - overlap_area
        
        return overlap_area / max(union_area, 1)
    
    def calculate_feature_similarity(self, features1, features2):
        return euclidean(features1, features2)
    
    def calculate_feature_correlation(self, features1, features2):
        correlation_matrix = np.corrcoef(features1, features2)
        return abs(correlation_matrix[0, 1]) if correlation_matrix.shape == (2, 2) else 0
    
    def evaluate_region_accuracy(self, features, labels, region_idx):
        from sklearn.ensemble import RandomForestClassifier
        
        if len(np.unique(labels)) < 2:
            return 0.5
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        try:
            clf.fit(features, labels)
            predictions = clf.predict(features)
            return accuracy_score(labels, predictions)
        except:
            return 0.5
    
    def update_sample_weights(self, weights, predictions, labels):
        new_weights = np.copy(weights)
        
        for i, (pred, true_label) in enumerate(zip(predictions, labels)):
            if pred == true_label:
                new_weights[i] *= np.exp(-2 * pred)
            else:
                new_weights[i] *= np.exp(1 - pred)
        
        new_weights /= np.sum(new_weights)
        return new_weights
    
    def select_cascade_features(self, image, labels=None):
        regions = self.generate_multiscale_regions(image.shape)
        
        if len(regions) == 0:
            return []
        
        all_features = self.feature_extractor.extract_features_from_regions(image, regions)
        
        if labels is None:
            labels = np.random.randint(0, 2, len(regions))
        
        selected_indices = []
        sample_weights = np.ones(len(regions)) / len(regions)
        
        best_region_idx = np.argmax([
            self.evaluate_region_accuracy(all_features[i:i+1], labels[i:i+1], i) 
            for i in range(len(regions))
        ])
        selected_indices.append(best_region_idx)
        
        while len(selected_indices) < min(20, len(regions)):
            best_candidate = -1
            best_score = -1
            
            for idx in range(len(regions)):
                if idx in selected_indices:
                    continue
                
                valid_candidate = True
                
                for selected_idx in selected_indices:
                    overlap = self.calculate_region_overlap(regions[idx], regions[selected_idx])
                    if overlap > 0.5:
                        valid_candidate = False
                        break
                    
                    similarity = self.calculate_feature_similarity(
                        all_features[idx], all_features[selected_idx]
                    )
                    if similarity < 0.3:
                        valid_candidate = False
                        break
                    
                    correlation = self.calculate_feature_correlation(
                        all_features[idx], all_features[selected_idx]
                    )
                    if correlation > 0.5:
                        valid_candidate = False
                        break
                
                if valid_candidate:
                    temp_indices = selected_indices + [idx]
                    temp_features = all_features[temp_indices]
                    temp_labels = labels[temp_indices]
                    
                    accuracy = self.evaluate_region_accuracy(temp_features, temp_labels, idx)
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_candidate = idx
            
            if best_candidate == -1:
                break
            
            selected_indices.append(best_candidate)
            
            selected_features = all_features[selected_indices]
            selected_labels = labels[selected_indices]
            predictions = np.random.randint(0, 2, len(selected_indices))
            
            sample_weights = self.update_sample_weights(
                sample_weights[selected_indices], predictions, selected_labels
            )
        
        self.selected_regions = [regions[i] for i in selected_indices]
        self.region_weights = sample_weights
        
        return self.selected_regions
    
    def apply_sliding_window(self, image, window_size=64):
        height, width = image.shape[:2]
        regions = []
        
        for y in range(0, height - window_size, self.step):
            for x in range(0, width - window_size, self.step):
                region = image[y:y+window_size, x:x+window_size]
                
                if np.mean(region) > 30:
                    regions.append((x, y, window_size, window_size))
        
        return regions
    
    def extract_features_from_selected_regions(self, image):
        if not self.selected_regions:
            return np.array([])
        
        return self.feature_extractor.extract_features_from_regions(image, self.selected_regions)