"""
explainability/gradcam.py
-------------------------
Implements Grad-CAM for ViT using pytorch-grad-cam.
"""

import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def reshape_transform(tensor, height=14, width=14):
    """
    Reshape transform specifically for ViT models.
    Removes the CLS token and reshapes the sequence of patch tokens
    back into a 2D spatial feature map.
    """
    # ViT output tensor shape: (B, num_tokens, D)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension like in CNNs
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def generate_gradcam_heatmap(
    model,
    image_tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray = None,
    image_size: int = 224,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a given target class.

    Args:
        model: Trained ChestViT model
        image_tensor: Normalized (1, 3, 224, 224) torch tensor
        target_class: Index of the disease to explain (0-13)
        original_image: Original RGB image in [0, 255] for overlay

    Returns:
        heatmap_overlay: (224, 224, 3) RGB image with Grad-CAM overlay
    """
    model.eval()

    # The target layer in Hugging Face ViT models is typically the last norm layer
    # before the classification head, or the final layer of the encoder.
    target_layers = [model.vit.encoder.layer[-1].layernorm_before]

    # Initialize GradCAM
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform=reshape_transform)

    # Generate the heatmap for the target class
    # target_category can be None to target the highest scoring class,
    # but we specify it for exact disease visualization
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(target_class)]

    # You can also pass aug_smooth=True and eigen_smooth=True
    grayscale_cam = cam(input_tensor=image_tensor,
                        targets=targets)

    grayscale_cam = grayscale_cam[0, :]

    if original_image is not None:
        # pytorch-grad-cam expects image float32 in [0, 1]
        rgb_img = np.float32(original_image) / 255
        # Resize to match output if needed
        if rgb_img.shape[:2] != (image_size, image_size):
            rgb_img = cv2.resize(rgb_img, (image_size, image_size))

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return visualization

    return grayscale_cam
