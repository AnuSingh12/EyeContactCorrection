
import torch
import torch.nn.functional as F


def preprocess_eye_images(images):
    """
    Preprocess grayscale synthetic eye images.

    Input:  [B, 1, H, W]
    Output: [B, 3, 32, 64]

    Note: Replicating grayscale into 3 channels does not
    create genuine RGB information.
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"Expected images shaped [B, 1, H, W], "
            f"got {tuple(images.shape)}"
        )

    images = images.float()

    images = F.interpolate(
        images,
        size=(32, 64),
        mode="bilinear",
        align_corners=False,
    )

    # Grayscale -> 3 identical channels.
    images = images.repeat(1, 3, 1, 1)

    return images


def prepare_target_angles(angles):
    """
    Validate two gaze-label components.

    Keep the original units unchanged until the dataset's
    angle convention and conversion to Cartesian components
    have been verified.
    """
    if angles.ndim != 2 or angles.shape[1] != 2:
        raise ValueError(
            f"Expected angles shaped [B, 2], "
            f"got {tuple(angles.shape)}"
        )

    return angles.float()
