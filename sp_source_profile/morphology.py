"""Morphology classification using a trained ResNet model.

Loads the cls_model.pth checkpoint and classifies single particle
SEM images into one of 10 morphology types.
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .resnet_model import initialize_resnet152

# Default morphology class names (from the paper listing order)
DEFAULT_CLASS_NAMES = ["Sp", "El", "MA", "RS", "Ro", "BS", "Cu", "Ir", "Ag", "CS"]

# Inference preprocessing transforms (same as sem_classifier test transforms)
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load a trained ResNet152 model from a checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to the .pth checkpoint file.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    nn.Module
        Loaded model in eval mode.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Determine number of classes

    model = initialize_resnet152(num_classes=10)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_class_names(ckpt_path: str) -> List[str]:
    """Retrieve class names from checkpoint or use defaults.

    Parameters
    ----------
    ckpt_path : str
        Path to the .pth checkpoint file.

    Returns
    -------
    list of str
        Class names ordered by model output index.
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "class_mapping" in checkpoint and isinstance(checkpoint["class_mapping"], dict):
            mapping = checkpoint["class_mapping"]
            # Ensure sorted by index
            sorted_indices = sorted(mapping.keys())
            return [mapping[i] for i in sorted_indices]
    except Exception:
        pass
    return DEFAULT_CLASS_NAMES


def classify_images(
    model: nn.Module,
    device: torch.device,
    pics_dir: str,
    part_numbers: List[int],
    class_names: Optional[List[str]] = None,
    batch_size: int = 64,
) -> Dict[int, str]:
    """Classify particle images by morphology.

    Parameters
    ----------
    model : nn.Module
        Trained ResNet model.
    device : torch.device
        Device for computation.
    pics_dir : str
        Directory containing particle images (png).
    part_numbers : list of int
        List of Part # values whose images need classification.
    class_names : list of str, optional
        Class name list indexed by model output. Defaults to DEFAULT_CLASS_NAMES.
    batch_size : int, optional
        Batch size for inference.

    Returns
    -------
    dict of {int: str}
        Mapping from Part # to morphology label.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    result: Dict[int, str] = {}
    transform = INFERENCE_TRANSFORMS

    # Build mapping from image filename to Part #
    # Image file naming pattern: {Part#}.png
    part_to_path = {}
    for pid in part_numbers:
        img_path = os.path.join(pics_dir, f"{pid}.png")
        if os.path.isfile(img_path):
            part_to_path[pid] = img_path
        else:
            result[pid] = "Unclassified"

    if not part_to_path:
        return result

    # Process in batches
    batch_pids = []
    batch_tensors = []

    sorted_pids = sorted(part_to_path.keys())
    for pid in tqdm(sorted_pids, desc="Classifying morphology"):
        img_path = part_to_path[pid]
        try:
            pil_img = Image.open(img_path)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            tensor = transform(pil_img)
            batch_pids.append(pid)
            batch_tensors.append(tensor)
        except Exception:
            result[pid] = "Unclassified"
            continue

        if len(batch_tensors) >= batch_size:
            _infer_batch(model, device, batch_pids, batch_tensors, class_names, result)
            batch_pids = []
            batch_tensors = []

    # Final batch
    if batch_tensors:
        _infer_batch(model, device, batch_pids, batch_tensors, class_names, result)

    return result


def _infer_batch(model, device, pids, tensors, class_names, result):
    """Run inference on a batch of tensors and update result dict."""
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
    for pid, pred_idx in zip(pids, preds.cpu().tolist()):
        if pred_idx < len(class_names):
            result[pid] = class_names[pred_idx]
        else:
            result[pid] = "Unclassified"
