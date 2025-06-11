import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

class DetectionMetrics:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        self.true_positives = []
        self.false_positives = []
        self.false_negatives = []
        self.confidences = []
        self.total_ground_truth = 0
    
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
    
    def update(self, predictions, ground_truth):
        pred_boxes = predictions.get('bboxes', [])
        pred_confidences = predictions.get('confidences', [])
        gt_boxes = ground_truth.get('bboxes', [])
        
        self.total_ground_truth += len(gt_boxes)
        
        matched_gt = set()
        
        for i, pred_box in enumerate(pred_boxes):
            confidence = pred_confidences[i] if i < len(pred_confidences) else 1.0
            self.confidences.append(confidence)
            
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j not in matched_gt:
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= self.iou_threshold:
                self.true_positives.append(1)
                self.false_positives.append(0)
                matched_gt.add(best_gt_idx)
            else:
                self.true_positives.append(0)
                self.false_positives.append(1)
        
        unmatched_gt = len(gt_boxes) - len(matched_gt)
        self.false_negatives.extend([1] * unmatched_gt)
    
    def calculate_precision_recall(self):
        if not self.true_positives:
            return 0.0, 0.0
        
        tp = np.sum(self.true_positives)
        fp = np.sum(self.false_positives)
        fn = np.sum(self.false_negatives)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    def calculate_f1_score(self):
        precision, recall = self.calculate_precision_recall()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_average_precision(self):
        if not self.true_positives or not self.confidences:
            return 0.0
        
        y_true = np.array(self.true_positives)
        y_scores = np.array(self.confidences)
        
        if len(np.unique(y_true)) < 2:
            return 0.0
        
        return average_precision_score(y_true, y_scores)
    
    def calculate_precision_recall_curve(self):
        if not self.true_positives or not self.confidences:
            return [], [], []
        
        y_true = np.array(self.true_positives)
        y_scores = np.array(self.confidences)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        return precision, recall, thresholds
    
    def get_metrics_summary(self):
        precision, recall = self.calculate_precision_recall()
        f1_score = self.calculate_f1_score()
        ap = self.calculate_average_precision()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'average_precision': ap,
            'total_predictions': len(self.true_positives),
            'total_ground_truth': self.total_ground_truth
        }

class SegmentationMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.intersection = 0
        self.union = 0
        self.total_pixels = 0
    
    def update(self, pred_mask, gt_mask, threshold=0.5):
        pred_binary = (pred_mask > threshold).astype(np.float32)
        gt_binary = gt_mask.astype(np.float32)
        
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(np.maximum(pred_binary, gt_binary))
        
        self.intersection += intersection
        self.union += union
        self.total_pixels += gt_binary.size
    
    def calculate_iou(self):
        if self.union == 0:
            return 1.0 if self.intersection == 0 else 0.0
        return self.intersection / self.union
    
    def calculate_dice(self):
        if self.intersection == 0 and self.union == 0:
            return 1.0
        return (2 * self.intersection) / (self.intersection + self.union)

def evaluate_model_performance(predictions_list, ground_truth_list, iou_threshold=0.5):
    detection_metrics = DetectionMetrics(iou_threshold)
    segmentation_metrics = SegmentationMetrics()
    
    for predictions, ground_truth in zip(predictions_list, ground_truth_list):
        detection_metrics.update(predictions, ground_truth)
        
        if 'segmentation' in predictions and 'mask' in ground_truth:
            segmentation_metrics.update(
                predictions['segmentation'], 
                ground_truth['mask']
            )
    
    detection_summary = detection_metrics.get_metrics_summary()
    segmentation_iou = segmentation_metrics.calculate_iou()
    segmentation_dice = segmentation_metrics.calculate_dice()
    
    return {
        'detection': detection_summary,
        'segmentation': {
            'iou': segmentation_iou,
            'dice': segmentation_dice
        }
    }