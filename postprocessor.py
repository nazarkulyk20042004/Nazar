import numpy as np
import cv2

class PostProcessor:
    def __init__(self, config):
        self.config = config['detection']
        self.confidence_threshold = self.config['confidence_threshold']
        self.nms_threshold = self.config['nms_threshold']
        self.max_detections = self.config['max_detections']
    
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def apply_nms(self, detections):
        if not detections:
            return []
        
        filtered_detections = [d for d in detections if d['confidence'] >= self.confidence_threshold]
        
        if not filtered_detections:
            return []
        
        filtered_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while filtered_detections:
            current = filtered_detections.pop(0)
            keep.append(current)
            
            remaining = []
            for detection in filtered_detections:
                iou = self.calculate_iou(current['bbox'], detection['bbox'])
                if iou < self.nms_threshold:
                    remaining.append(detection)
            
            filtered_detections = remaining
        
        return keep[:self.max_detections]
    
    def filter_by_size(self, detections, min_size=20, max_size=500):
        filtered = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            width = x2 - x1
            height = y2 - y1
            
            if min_size <= width <= max_size and min_size <= height <= max_size:
                filtered.append(detection)
        
        return filtered
    
    def merge_overlapping_detections(self, detections, overlap_threshold=0.3):
        if len(detections) <= 1:
            return detections
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self.calculate_iou(det1['bbox'], det2['bbox'])
                if iou > overlap_threshold:
                    group.append(det2)
                    used.add(j)
            
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged_detections = self.merge_overlapping_detections(filtered_detections)
        
        nms_detections = self.apply_nms(merged_detections)
        
        return nms_detections
    
    def extract_final_bboxes(self, detections):
        bboxes = []
        confidences = []
        
        for detection in detections:
            bboxes.append(detection['bbox'])
            confidences.append(detection['confidence'])
        
        return np.array(bboxes), np.array(confidences)
    
    def convert_to_coco_format(self, detections):
        coco_detections = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            width = x2 - x1
            height = y2 - y1
            
            coco_detection = {
                'bbox': [x1, y1, width, height],
                'score': detection['confidence'],
                'category_id': 1
            }
            coco_detections.append(coco_detection)
        
        return coco_detectionsdetection = self.merge_detection_group(group)
                merged.append(merged_detection)
        
        return merged
    
    def merge_detection_group(self, group):
        total_confidence = sum(det['confidence'] for det in group)
        weights = [det['confidence'] / total_confidence for det in group]
        
        avg_bbox = [0, 0, 0, 0]
        for i, det in enumerate(group):
            weight = weights[i]
            bbox = det['bbox']
            for j in range(4):
                avg_bbox[j] += bbox[j] * weight
        
        avg_confidence = total_confidence / len(group)
        
        return {
            'bbox': tuple(map(int, avg_bbox)),
            'confidence': avg_confidence,
            'segmentation': group[0]['segmentation'],
            'region_confidence': group[0]['region_confidence']
        }
    
    def adaptive_threshold_adjustment(self, detections, image_stats):
        mean_intensity = image_stats.get('mean_intensity', 128)
        
        if mean_intensity < 50:
            adjusted_threshold = self.confidence_threshold * 0.8
        elif mean_intensity < 100:
            adjusted_threshold = self.confidence_threshold * 0.9
        else:
            adjusted_threshold = self.confidence_threshold
        
        return [det for det in detections if det['confidence'] >= adjusted_threshold]
    
    def post_process(self, detections, image_stats=None):
        if not detections:
            return []
        
        filtered_detections = self.filter_by_size(detections)
        
        if image_stats:
            filtered_detections = self.adaptive_threshold_adjustment(filtered_detections, image_stats)
        
        merged_