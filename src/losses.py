
import torch
import torch.nn.functional as F


def correction_loss(corrected_image, target_image):
    """
    Lc: Mean squared error between corrected and target images.
    """
    if corrected_image.shape != target_image.shape:
        raise ValueError(
            "Corrected image and target image must have the same shape."
        )

    return F.mse_loss(corrected_image, target_image)


def reconstruction_loss(reconstructed_image, original_image):
    """
    Lr: Mean squared error between reconstructed and original images.
    """
    if reconstructed_image.shape != original_image.shape:
        raise ValueError(
            "Reconstructed image and original image must have the same shape."
        )

    return F.mse_loss(reconstructed_image, original_image)


def total_loss(lc, lr):
    """
    Paper's combined objective: 0.8 * Lc + 0.2 * Lr.
    """
    return 0.8 * lc + 0.2 * lr
