import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def plot_training_history(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    train_acc = [1 - loss for loss in train_losses]
    val_acc = [1 - loss for loss in val_losses]
    plt.plot(train_acc, label='Training Accuracy', color='blue')
    plt.plot(val_acc, label='Validation Accuracy', color='red', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_detection_results(image, detections, save_path=None):
    if len(image.shape) == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(display_image)
    plt.title('Vehicle Detection Results')
    plt.axis('off')
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        rect = plt.Rectangle((x1, y1), width, height, 
                           linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        
        plt.text(x1, y1 - 5, f'Vehicle: {confidence:.2f}', 
                color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_metrics_comparison(metrics_dict, save_path=None):
    categories = list(metrics_dict.keys())
    ap_scores = [metrics_dict[cat].get('average_precision', 0) for cat in categories]
    recall_scores = [metrics_dict[cat].get('recall', 0) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, ap_scores, width, label='AP@0.5', color='skyblue')
    plt.bar(x + width/2, recall_scores, width, label='Recall', color='orange')
    
    plt.title('Detection Performance by Object Type')
    plt.xlabel('Object Type')
    plt.ylabel('Score')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, (ap, recall) in enumerate(zip(ap_scores, recall_scores)):
        plt.text(i - width/2, ap + 0.01, f'{ap:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, recall + 0.01, f'{recall:.3f}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_illumination_analysis(illumination_levels, accuracy_scores, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.bar(illumination_levels, accuracy_scores, color=['darkblue', 'blue', 'lightblue'])
    plt.title('Detection Accuracy vs Illumination Level')
    plt.xlabel('Illumination Level')
    plt.ylabel('AP@0.5')
    plt.ylim(0, 1)
    
    for i, score in enumerate(accuracy_scores):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_precision_recall_curve(precision, recall, average_precision, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', linewidth=2, 
             label=f'AP = {average_precision:.3f}')
    plt.fill_between(recall, precision, alpha=0.3, color='blue')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def save_detection_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    image_results = results.get('images', [])
    metrics = results.get('metrics', {})
    
    for i, result in enumerate(image_results):
        image = result['image']
        detections = result['detections']
        image_path = os.path.join(output_dir, f'detection_{i:04d}.png')
        plot_detection_results(image, detections, image_path)
    
    if 'training_history' in results:
        history = results['training_history']
        history_path = os.path.join(output_dir, 'training_history.png')
        plot_training_history(history['train_loss'], history['val_loss'], history_path)
    
    if 'detection_metrics' in metrics:
        detection_metrics = metrics['detection_metrics']
        
        if 'precision_recall_curve' in detection_metrics:
            pr_data = detection_metrics['precision_recall_curve']
            pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
            plot_precision_recall_curve(pr_data['precision'], pr_data['recall'], 
                                       pr_data['average_precision'], pr_path)
        
        if 'object_comparison' in detection_metrics:
            comparison_data = detection_metrics['object_comparison']
            comparison_path = os.path.join(output_dir, 'object_comparison.png')
            plot_metrics_comparison(comparison_data, comparison_path)
        
        if 'illumination_analysis' in detection_metrics:
            illumination_data = detection_metrics['illumination_analysis']
            illumination_path = os.path.join(output_dir, 'illumination_analysis.png')
            plot_illumination_analysis(illumination_data['levels'], 
                                     illumination_data['scores'], illumination_path)

def create_detection_summary_report(results, output_path):
    metrics = results.get('metrics', {})
    
    with open(output_path, 'w') as f:
        f.write("Night Vehicle Detection Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        if 'detection' in metrics:
            det_metrics = metrics['detection']
            f.write("Detection Metrics:\n")
            f.write(f"Precision: {det_metrics.get('precision', 0):.3f}\n")
            f.write(f"Recall: {det_metrics.get('recall', 0):.3f}\n")
            f.write(f"F1-Score: {det_metrics.get('f1_score', 0):.3f}\n")
            f.write(f"Average Precision: {det_metrics.get('average_precision', 0):.3f}\n")
            f.write(f"Total Predictions: {det_metrics.get('total_predictions', 0)}\n")
            f.write(f"Total Ground Truth: {det_metrics.get('total_ground_truth', 0)}\n\n")
        
        if 'segmentation' in metrics:
            seg_metrics = metrics['segmentation']
            f.write("Segmentation Metrics:\n")
            f.write(f"IoU: {seg_metrics.get('iou', 0):.3f}\n")
            f.write(f"Dice Score: {seg_metrics.get('dice', 0):.3f}\n\n")
        
        f.write("Configuration Used:\n")
        config = results.get('config', {})
        for section, params in config.items():
            f.write(f"{section}:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")