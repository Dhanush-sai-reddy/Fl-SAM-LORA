"""
LoRA (Low-Rank Adaptation) implementation for SAM (Segment Anything Model)
"""
import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps around a linear layer.
    Implements: h = W_0 * x + (B * A) * x * scaling
    where W_0 is frozen, B and A are low-rank matrices
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
            original_output: output from the frozen linear layer
        Returns:
            combined output: original_output + LoRA adaptation
        """
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return original_output + lora_out


def inject_lora_into_sam(
    sam_model,
    target_modules: list = ["qkv"],
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.1,
):
    """
    Inject LoRA layers into SAM's vision transformer.
    
    Args:
        sam_model: SAM model instance
        target_modules: list of module names to add LoRA to (e.g., ["qkv", "proj"])
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: dropout rate
    """
    lora_layers = {}
    
    # Access the image encoder (ViT)
    image_encoder = sam_model.image_encoder
    
    # Iterate through all blocks
    for block_idx, block in enumerate(image_encoder.blocks):
        # Add LoRA to attention qkv projection
        if "qkv" in target_modules and hasattr(block.attn, "qkv"):
            qkv = block.attn.qkv
            if isinstance(qkv, nn.Linear):
                lora_layer = LoRALayer(
                    qkv.in_features,
                    qkv.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                lora_layers[f"block_{block_idx}_qkv"] = lora_layer
                
                # Freeze original qkv
                qkv.weight.requires_grad = False
                if qkv.bias is not None:
                    qkv.bias.requires_grad = False
                
                # Replace forward to include LoRA
                original_forward = qkv.forward
                def make_lora_forward(orig_fwd, lora):
                    def forward(x):
                        orig_out = orig_fwd(x)
                        return lora(x, orig_out)
                    return forward
                qkv.forward = make_lora_forward(original_forward, lora_layer)
        
        # Add LoRA to attention output projection
        if "proj" in target_modules and hasattr(block.attn, "proj"):
            proj = block.attn.proj
            if isinstance(proj, nn.Linear):
                lora_layer = LoRALayer(
                    proj.in_features,
                    proj.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                lora_layers[f"block_{block_idx}_proj"] = lora_layer
                
                # Freeze original proj
                proj.weight.requires_grad = False
                if proj.bias is not None:
                    proj.bias.requires_grad = False
                
                # Replace forward to include LoRA
                original_forward = proj.forward
                def make_lora_forward(orig_fwd, lora):
                    def forward(x):
                        orig_out = orig_fwd(x)
                        return lora(x, orig_out)
                    return forward
                proj.forward = make_lora_forward(original_forward, lora_layer)
        
        # Add LoRA to MLP layers if specified
        if "mlp" in target_modules and hasattr(block, "mlp"):
            for mlp_idx, layer in enumerate([block.mlp.lin1, block.mlp.lin2]):
                if isinstance(layer, nn.Linear):
                    lora_layer = LoRALayer(
                        layer.in_features,
                        layer.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    lora_layers[f"block_{block_idx}_mlp_{mlp_idx}"] = lora_layer
                    
                    # Freeze original layer
                    layer.weight.requires_grad = False
                    if layer.bias is not None:
                        layer.bias.requires_grad = False
                    
                    # Replace forward to include LoRA
                    original_forward = layer.forward
                    def make_lora_forward(orig_fwd, lora):
                        def forward(x):
                            orig_out = orig_fwd(x)
                            return lora(x, orig_out)
                        return forward
                    layer.forward = make_lora_forward(original_forward, lora_layer)
    
    # Register LoRA layers as a ModuleDict in the model
    sam_model.lora_layers = nn.ModuleDict(lora_layers)
    
    return sam_model


def get_lora_parameters(sam_model):
    """Extract only LoRA parameters for optimization"""
    if hasattr(sam_model, "lora_layers"):
        return list(sam_model.lora_layers.parameters())
    return []


def save_lora_weights(sam_model, save_path: str):
    """Save only LoRA weights"""
    if hasattr(sam_model, "lora_layers"):
        torch.save(sam_model.lora_layers.state_dict(), save_path)
    else:
        raise ValueError("Model does not have LoRA layers")


def load_lora_weights(sam_model, load_path: str, device: str = "cpu"):
    """Load LoRA weights"""
    if hasattr(sam_model, "lora_layers"):
        state_dict = torch.load(load_path, map_location=device)
        sam_model.lora_layers.load_state_dict(state_dict)
    else:
        raise ValueError("Model does not have LoRA layers")
