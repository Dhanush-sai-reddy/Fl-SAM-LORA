import torch
import os
from typing import Optional


def init_sam_with_lora(
    model_type: str = "vit_b",
    sam_checkpoint: Optional[str] = None,
    lora_weights: str = None,
    device: Optional[str] = None,
):
    from segment_anything import sam_model_registry, SamPredictor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    if not lora_weights or not os.path.isfile(lora_weights):
        raise FileNotFoundError("lora_weights path must be provided and exist for SAM+LoRA initialization")

    sd = torch.load(lora_weights, map_location=device)
    sam.load_state_dict(sd, strict=False)

    predictor = SamPredictor(sam)
    return sam, predictor
