import torch
import os
import cv2
from tqdm import tqdm
import numpy as np
from src.data.data_loader import create_dataloaders
from src.utils.metrics import DetectionMetrics
from src.utils.visualization import plot_training_history

def train_epoch(detector, train_loader, epoch):
    detector.set_model_mode('train')
    
    total_loss = 0
    dice_loss = 0
    focal_loss = 0
    boundary_loss = 0
    iou_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        losses = detector.train_step(batch)
        
        total_loss += losses['total_loss'].item()
        dice_loss += losses['dice_loss'].item()
        focal_loss += losses['focal_loss'].item()
        boundary_loss += losses['boundary_loss'].item()
        iou_loss += losses['iou_loss'].item()
        
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return {
        'total_loss': total_loss / len(train_loader),
        'dice_loss': dice_loss / len(train_loader),
        'focal_loss': focal_loss / len(train_loader),
        'boundary_loss': boundary_loss / len(train_loader),
        'iou_loss': iou_loss / len(train_loader)
    }

def validate_epoch(detector, val_loader, epoch):
    detector.set_model_mode('eval')
    
    total_loss = 0
    dice_loss = 0
    focal_loss = 0
    boundary_loss = 0
    iou_loss = 0
    
    metrics = DetectionMetrics()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation {epoch}'):
            losses, predictions = detector.validation_step(batch)
            
            total_loss += losses['total_loss'].item()
            dice_loss += losses['dice_loss'].item()
            focal_loss += losses['focal_loss'].item()
            boundary_loss += losses['boundary_loss'].item()
            iou_loss += losses['iou_loss'].item()
            
            batch_size = batch['image'].size(0)
            for i in range(batch_size):
                pred_result = extract_predictions(predictions, i)
                gt_result = extract_ground_truth(batch, i)
                metrics.update(pred_result, gt_result)
    
    metrics_summary = metrics.get_metrics_summary()
    
    return {
        'total_loss': total_loss / len(val_loader),
        'dice_loss': dice_loss / len(val_loader),
        'focal_loss': focal_loss / len(val_loader),
        'boundary_loss': boundary_loss / len(val_loader),
        'iou_loss': iou_loss / len(val_loader),
        'metrics': metrics_summary
    }

def extract_predictions(predictions, batch_idx):
    seg = predictions['segmentation'][batch_idx].cpu().numpy()
    conf = predictions['confidence'][batch_idx].cpu().numpy()
    bbox_reg = predictions['bbox_regression'][batch_idx].cpu().numpy()
    
    threshold = 0.5
    binary_mask = seg > threshold
    
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    bboxes = []
    confidences = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x + w, y + h])
            
            mask_region = binary_mask[y:y+h, x:x+w]
            region_conf = np.mean(conf[y:y+h, x:x+w]) if mask_region.any() else 0
            confidences.append(region_conf)
    
    return {
        'bboxes': bboxes,
        'confidences': confidences
    }

def extract_ground_truth(batch, batch_idx):
    bboxes = batch['bboxes'][batch_idx].cpu().numpy()
    
    valid_bboxes = []
    for bbox in bboxes:
        if np.sum(bbox) > 0:
            valid_bboxes.append(bbox.tolist())
    
    return {
        'bboxes': valid_bboxes
    }

def train_model(detector, config):
    train_loader, val_loader, _ = create_dataloaders(config)
    
    epochs = config['training']['epochs']
    output_dir = config.get('output_dir', 'results/training')
    os.makedirs(output_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_results = train_epoch(detector, train_loader, epoch)
        val_results = validate_epoch(detector, val_loader, epoch)
        
        detector.scheduler.step()
        
        train_loss = train_results['total_loss']
        val_loss = val_results['total_loss']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Precision: {val_results['metrics']['precision']:.4f}")
        print(f"Val Recall: {val_results['metrics']['recall']:.4f}")
        print(f"Val F1: {val_results['metrics']['f1_score']:.4f}")
        print("-" * 50)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            detector.save_model(best_model_path)
        
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            detector.save_model(checkpoint_path)
    
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    detector.save_model(final_model_path)
    
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, history_plot_path)
    
    training_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_metrics': val_results['metrics']
    }
    
    return training_results