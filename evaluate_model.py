import torch
import numpy as np
import cv2
from tqdm import tqdm
from src.data.data_loader import create_dataloaders
from src.utils.metrics import DetectionMetrics, evaluate_model_performance
from src.utils.visualization import plot_metrics_comparison, plot_illumination_analysis

def evaluate_by_object_type(detector, test_loader):
    object_metrics = {
        'vehicles': DetectionMetrics(),
        'headlights': DetectionMetrics(),
        'reflections': DetectionMetrics()
    }
    
    detector.set_model_mode('eval')
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating by object type'):
            batch_size = batch['image'].size(0)
            
            for i in range(batch_size):
                image = batch['image'][i:i+1]
                result = detector.detect(image.squeeze().cpu().numpy())
                
                detections = result['detections']
                ground_truth = extract_ground_truth_with_types(batch, i)
                
                for obj_type in object_metrics.keys():
                    type_detections = filter_detections_by_type(detections, obj_type)
                    type_gt = filter_ground_truth_by_type(ground_truth, obj_type)
                    
                    object_metrics[obj_type].update(
                        {'bboxes': [d['bbox'] for d in type_detections],
                         'confidences': [d['confidence'] for d in type_detections]},
                        {'bboxes': type_gt}
                    )
    
    results = {}
    for obj_type, metrics in object_metrics.items():
        results[obj_type] = metrics.get_metrics_summary()
    
    return results

def evaluate_by_illumination_level(detector, test_loader):
    illumination_metrics = {
        'low': DetectionMetrics(),
        'medium': DetectionMetrics(),
        'high': DetectionMetrics()
    }
    
    detector.set_model_mode('eval')
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating by illumination'):
            batch_size = batch['image'].size(0)
            
            for i in range(batch_size):
                image_tensor = batch['image'][i:i+1]
                image_np = image_tensor.squeeze().cpu().numpy()
                
                illumination_level = classify_illumination_level(image_np)
                
                result = detector.detect(image_np)
                detections = result['detections']
                ground_truth = extract_ground_truth_simple(batch, i)
                
                illumination_metrics[illumination_level].update(
                    {'bboxes': [d['bbox'] for d in detections],
                     'confidences': [d['confidence'] for d in detections]},
                    {'bboxes': ground_truth}
                )
    
    results = {}
    for level, metrics in illumination_metrics.items():
        results[level] = metrics.get_metrics_summary()
    
    return results

def classify_illumination_level(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mean_intensity = np.mean(image)
    
    if mean_intensity < 60:
        return 'low'
    elif mean_intensity < 120:
        return 'medium'
    else:
        return 'high'

def extract_ground_truth_with_types(batch, batch_idx):
    bboxes = batch['bboxes'][batch_idx].cpu().numpy()
    
    ground_truth = {
        'vehicles': [],
        'headlights': [],
        'reflections': []
    }
    
    for bbox in bboxes:
        if np.sum(bbox) > 0:
            object_type = classify_object_type(bbox)
            ground_truth[object_type].append(bbox.tolist())
    
    return ground_truth

def extract_ground_truth_simple(batch, batch_idx):
    bboxes = batch['bboxes'][batch_idx].cpu().numpy()
    
    valid_bboxes = []
    for bbox in bboxes:
        if np.sum(bbox) > 0:
            valid_bboxes.append(bbox.tolist())
    
    return valid_bboxes

def classify_object_type(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    if area > 5000:
        return 'vehicles'
    elif area > 500:
        return 'headlights'
    else:
        return 'reflections'

def filter_detections_by_type(detections, obj_type):
    filtered = []
    
    for detection in detections:
        bbox = detection['bbox']
        detected_type = classify_object_type(bbox)
        
        if detected_type == obj_type:
            filtered.append(detection)
    
    return filtered

def filter_ground_truth_by_type(ground_truth, obj_type):
    return ground_truth.get(obj_type, [])

def comprehensive_evaluation(detector, test_loader):
    overall_metrics = DetectionMetrics()
    all_predictions = []
    all_ground_truth = []
    
    detector.set_model_mode('eval')
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Comprehensive evaluation'):
            batch_size = batch['image'].size(0)
            
            for i in range(batch_size):
                image_tensor = batch['image'][i:i+1]
                image_np = image_tensor.squeeze().cpu().numpy()
                
                result = detector.detect(image_np)
                detections = result['detections']
                ground_truth = extract_ground_truth_simple(batch, i)
                
                predictions = {
                    'bboxes': [d['bbox'] for d in detections],
                    'confidences': [d['confidence'] for d in detections]
                }
                
                gt = {'bboxes': ground_truth}
                
                overall_metrics.update(predictions, gt)
                all_predictions.append(predictions)
                all_ground_truth.append(gt)
    
    overall_results = overall_metrics.get_metrics_summary()
    
    precision, recall, thresholds = overall_metrics.calculate_precision_recall_curve()
    
    return {
        'overall': overall_results,
        'precision_recall_curve': {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': overall_results['average_precision']
        },
        'predictions': all_predictions,
        'ground_truth': all_ground_truth
    }

def evaluate_model(detector, config, model_path=None):
    if model_path:
        detector.load_model(model_path)
    
    _, _, test_loader = create_dataloaders(config)
    
    print("Running comprehensive evaluation...")
    comprehensive_results = comprehensive_evaluation(detector, test_loader)
    
    print("Evaluating by object type...")
    object_type_results = evaluate_by_object_type(detector, test_loader)
    
    print("Evaluating by illumination level...")
    illumination_results = evaluate_by_illumination_level(detector, test_loader)
    
    print("\nOverall Results:")
    overall = comprehensive_results['overall']
    print(f"Precision: {overall['precision']:.3f}")
    print(f"Recall: {overall['recall']:.3f}")
    print(f"F1-Score: {overall['f1_score']:.3f}")
    print(f"Average Precision: {overall['average_precision']:.3f}")
    
    print("\nResults by Object Type:")
    for obj_type, metrics in object_type_results.items():
        print(f"{obj_type.capitalize()}:")
        print(f"  AP: {metrics['average_precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
    
    print("\nResults by Illumination Level:")
    for level, metrics in illumination_results.items():
        print(f"{level.capitalize()} illumination:")
        print(f"  AP: {metrics['average_precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
    
    evaluation_results = {
        'comprehensive': comprehensive_results,
        'object_type': object_type_results,
        'illumination': illumination_results,
        'config': config
    }
    
    return evaluation_results