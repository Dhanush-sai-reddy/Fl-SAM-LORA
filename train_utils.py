"""
Training utilities for SAM with LoRA on Fed-KITS dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader


class KITSDataset(Dataset):
    """Dataset for KITS medical imaging data"""
    def __init__(self, items: List[Dict[str, np.ndarray]], transform=None):
        """
        Args:
            items: List of dictionaries with 'image' and optionally 'label' keys
            transform: Optional transforms
        """
        self.items = items
        self.transform = transform
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image = item["image"]  # Should be 2D or 3D numpy array
        
        # Convert to 3-channel if grayscale
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=0)  # (3, H, W)
        elif image.ndim == 3:
            # Assume (H, W, D) format, take middle slice and replicate
            mid_slice = image[:, :, image.shape[2] // 2]
            image = np.stack([mid_slice, mid_slice, mid_slice], axis=0)
        
        # Normalize to [0, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Resize to SAM input size (1024x1024)
        if image.shape[1] != 1024 or image.shape[2] != 1024:
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(1024, 1024), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Get label if available
        if "label" in item and item["label"] is not None:
            label = item["label"]
            if label.ndim == 2:
                label = torch.from_numpy(label).long()
            elif label.ndim == 3:
                mid_slice = label[:, :, label.shape[2] // 2]
                label = torch.from_numpy(mid_slice).long()
            
            # Resize label
            if label.shape[0] != 1024 or label.shape[1] != 1024:
                label = F.interpolate(
                    label.unsqueeze(0).unsqueeze(0).float(),
                    size=(1024, 1024),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            
            # Binary segmentation (background vs kidney+tumor)
            label = (label > 0).long()
        else:
            label = torch.zeros((1024, 1024), dtype=torch.long)
        
        return {"image": image, "label": label}


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute Dice loss for segmentation.
    
    Args:
        pred: Predicted masks (B, H, W) with values in [0, 1]
        target: Ground truth masks (B, H, W) with binary values
        smooth: Smoothing factor
    """
    pred = pred.flatten(1)
    target = target.flatten(1).float()
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """
    Combined BCE + Dice loss
    
    Args:
        pred: Predicted masks (B, H, W) with values in [0, 1]
        target: Ground truth masks (B, H, W) with binary values
        bce_weight: Weight for BCE loss
    """
    bce = F.binary_cross_entropy(pred, target.float())
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics: Dice coefficient and IoU
    
    Args:
        pred: Predicted masks (B, H, W) with values in [0, 1]
        target: Ground truth masks (B, H, W) with binary values
    """
    pred_binary = (pred > 0.5).float()
    target = target.float()
    
    # Dice coefficient
    pred_flat = pred_binary.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    
    # IoU
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = ((pred_flat + target_flat) > 0).float().sum(dim=1)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
    }


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Train for one epoch
    
    Args:
        model: SAM model with LoRA
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        max_steps: Maximum number of steps (for quick testing)
    
    Returns:
        Dictionary with average loss and metrics
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    for step, batch in enumerate(dataloader):
        if max_steps and step >= max_steps:
            break
        
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass through image encoder
        with torch.no_grad():
            image_embeddings = model.image_encoder(images)
        
        # Simple segmentation head (for demonstration)
        # In practice, you'd use SAM's mask decoder with proper prompts
        # Here we'll use a simple upsampling approach
        B, C, H, W = image_embeddings.shape
        
        # Upsample embeddings to mask size
        pred_masks = F.interpolate(
            image_embeddings,
            size=(1024, 1024),
            mode='bilinear',
            align_corners=False
        )
        
        # Reduce channels to 1 (simple projection)
        pred_masks = pred_masks.mean(dim=1, keepdim=True).squeeze(1)  # (B, H, W)
        pred_masks = torch.sigmoid(pred_masks)
        
        # Compute loss
        loss = combined_loss(pred_masks, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(pred_masks, labels)
        
        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        num_batches += 1
    
    if num_batches == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    
    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }


def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate the model
    
    Args:
        model: SAM model with LoRA
        dataloader: Validation data loader
        device: Device to evaluate on
        max_steps: Maximum number of steps
    
    Returns:
        Dictionary with average loss and metrics
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_steps and step >= max_steps:
                break
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            image_embeddings = model.image_encoder(images)
            
            # Simple segmentation head
            B, C, H, W = image_embeddings.shape
            pred_masks = F.interpolate(
                image_embeddings,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )
            pred_masks = pred_masks.mean(dim=1, keepdim=True).squeeze(1)
            pred_masks = torch.sigmoid(pred_masks)
            
            # Compute loss and metrics
            loss = combined_loss(pred_masks, labels)
            metrics = compute_metrics(pred_masks, labels)
            
            total_loss += loss.item()
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            num_batches += 1
    
    if num_batches == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    
    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }
