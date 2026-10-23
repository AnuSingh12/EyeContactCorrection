
import torch
import torch.nn.functional as F


def warp_image(image, vector_field):
    """
    Warp an image using a 2-channel displacement field.

    image:        [B, C, H, W]
    vector_field: [B, 2, H, W]
                  channel 0 = horizontal displacement (dx)
                  channel 1 = vertical displacement (dy)
    """
    if image.ndim != 4 or vector_field.ndim != 4:
        raise ValueError("Image and vector field must be 4D tensors.")

    batch_size, _, height, width = image.shape

    if vector_field.shape != (batch_size, 2, height, width):
        raise ValueError(
            "Vector field must have shape [B, 2, H, W]."
        )

    y = torch.linspace(
        -1, 1, height, device=image.device, dtype=image.dtype
    )
    x = torch.linspace(
        -1, 1, width, device=image.device, dtype=image.dtype
    )

    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    base_grid = torch.stack(
        (grid_x, grid_y), dim=-1
    ).unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Convert pixel displacement to normalized coordinates.
    dx = vector_field[:, 0] * (2.0 / max(width - 1, 1))
    dy = vector_field[:, 1] * (2.0 / max(height - 1, 1))

    displacement = torch.stack((dx, dy), dim=-1)

    # Diagnostic candidate: reverse sampling displacement.
    sampling_grid = base_grid + displacement

    return F.grid_sample(
        image,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def apply_brightness_map(warped_image, brightness_map):
    """
    Blend the warped image toward white using the predicted mask.
    This is the current implementation choice, not a confirmed
    reproduction of the paper's exact brightness equation.
    """
    if brightness_map.shape != (
        warped_image.shape[0],
        1,
        warped_image.shape[2],
        warped_image.shape[3],
    ):
        raise ValueError(
            "Brightness map must have shape [B, 1, H, W]."
        )

    return warped_image * (1.0 - brightness_map) + brightness_map


def correct_eye(image, vector_field, brightness_map):
    """Warp the eye image and apply its predicted brightness map."""
    warped = warp_image(image, vector_field)
    corrected = apply_brightness_map(warped, brightness_map)
    return corrected
