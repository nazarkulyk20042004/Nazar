import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        return self.mse(pred_edges, target_edges)
    
    def _compute_edges(self, tensor):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).to(tensor.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(tensor.device)
        
        edges_x = F.conv2d(tensor, sobel_x, padding=1)
        edges_y = F.conv2d(tensor, sobel_y, padding=1)
        
        return torch.sqrt(edges_x**2 + edges_y**2)

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred_boxes, target_boxes):
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(dim=-1)
        
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        union_area = pred_area + target_area - inter_area
        iou = (inter_area + self.smooth) / (union_area + self.smooth)
        
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        weights = config['training']['loss_weights']
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
        self.iou_loss = IoULoss()
        
        self.weights = {
            'dice': weights['dice'],
            'focal': weights['focal'],
            'boundary': weights['boundary'],
            'box': weights['box']
        }
    
    def forward(self, predictions, targets):
        pred_seg = predictions['segmentation']
        pred_bbox = predictions['bbox_regression']
        pred_conf = predictions['confidence']
        
        target_mask = targets['mask']
        target_bbox = targets['bboxes']
        
        dice_loss = self.dice_loss(pred_seg, target_mask)
        focal_loss = self.focal_loss(pred_conf, target_mask)
        boundary_loss = self.boundary_loss(pred_seg, target_mask)
        
        if target_bbox.numel() > 0:
            iou_loss = self.iou_loss(pred_bbox.view(-1, 4), target_bbox.view(-1, 4))
        else:
            iou_loss = torch.tensor(0.0, device=pred_seg.device)
        
        total_loss = (
            self.weights['dice'] * dice_loss +
            self.weights['focal'] * focal_loss +
            self.weights['boundary'] * boundary_loss +
            self.weights['box'] * iou_loss
        )
        
        return {
            'total_loss': total_loss,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss,
            'boundary_loss': boundary_loss,
            'iou_loss': iou_loss
        }