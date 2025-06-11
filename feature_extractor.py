import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern

class HOGFeatureExtractor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=pixels_per_cell,
            _nbins=orientations
        )
    
    def extract(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, (64, 64))
        features = self.hog.compute(image)
        return features.flatten()[:18]

class LBPFeatureExtractor:
    def __init__(self, radius=3, n_points=24):
        self.radius = radius
        self.n_points = n_points
    
    def extract(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lbp = local_binary_pattern(image, self.n_points, self.radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=self.n_points + 2, range=(0, self.n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        return hist[:32]

class DirectionalFeatureExtractor:
    def __init__(self):
        self.kernels = self._create_directional_kernels()
    
    def _create_directional_kernels(self):
        kernels = []
        angles = [0, 45, 90, 135]
        
        for angle in angles:
            kernel = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=np.float32)
            
            if angle == 45:
                kernel = np.array([
                    [0, 1, 2],
                    [-1, 0, 1],
                    [-2, -1, 0]
                ], dtype=np.float32)
            elif angle == 90:
                kernel = np.array([
                    [-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]
                ], dtype=np.float32)
            elif angle == 135:
                kernel = np.array([
                    [-2, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 2]
                ], dtype=np.float32)
            
            kernels.append(kernel)
        
        return kernels
    
    def extract(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        responses = []
        for kernel in self.kernels:
            response = cv2.filter2D(image, -1, kernel)
            responses.append(np.mean(np.abs(response)))
        
        return np.array(responses[:8])

class ContrastFeatureExtractor:
    def __init__(self):
        self.light_threshold = 200
        self.min_contour_area = 50
    
    def extract(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        light_regions = self._detect_light_regions(image)
        geometric_features = self._analyze_light_geometry(light_regions, image.shape)
        intensity_features = self._analyze_light_intensity(image, light_regions)
        
        return np.concatenate([geometric_features, intensity_features])
    
    def _detect_light_regions(self, image):
        _, binary = cv2.threshold(image, self.light_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                valid_contours.append(contour)
        
        return valid_contours
    
    def _analyze_light_geometry(self, contours, image_shape):
        if not contours:
            return np.zeros(4)
        
        features = []
        total_area = sum(cv2.contourArea(c) for c in contours)
        features.append(total_area / (image_shape[0] * image_shape[1]))
        features.append(len(contours))
        
        if len(contours) >= 2:
            centers = [self._get_contour_center(c) for c in contours]
            distances = [np.linalg.norm(np.array(centers[i]) - np.array(centers[j])) 
                        for i in range(len(centers)) for j in range(i+1, len(centers))]
            features.append(np.mean(distances) if distances else 0)
        else:
            features.append(0)
        
        aspect_ratios = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratios.append(w / max(h, 1))
        features.append(np.mean(aspect_ratios) if aspect_ratios else 0)
        
        return np.array(features)
    
    def _analyze_light_intensity(self, image, contours):
        if not contours:
            return np.zeros(2)
        
        intensities = []
        for contour in contours:
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            mean_intensity = cv2.mean(image, mask=mask)[0]
            intensities.append(mean_intensity)
        
        return np.array([np.mean(intensities), np.std(intensities)])
    
    def _get_contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return (0, 0)

class FeatureExtractor:
    def __init__(self):
        self.hog_extractor = HOGFeatureExtractor()
        self.lbp_extractor = LBPFeatureExtractor()
        self.directional_extractor = DirectionalFeatureExtractor()
        self.contrast_extractor = ContrastFeatureExtractor()
    
    def extract_all_features(self, image):
        hog_features = self.hog_extractor.extract(image)
        lbp_features = self.lbp_extractor.extract(image)
        directional_features = self.directional_extractor.extract(image)
        contrast_features = self.contrast_extractor.extract(image)
        
        return np.concatenate([hog_features, lbp_features, directional_features, contrast_features])
    
    def extract_features_from_regions(self, image, regions):
        features_list = []
        for region in regions:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            features = self.extract_all_features(roi)
            features_list.append(features)
        
        return np.array(features_list)