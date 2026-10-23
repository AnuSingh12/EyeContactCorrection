
import torch
import torch.nn.functional as F


def warp_image(image, vector_field):
    """
    Warp an image using a predicted displacement field.

    image:
        [B, C, H, W], normalized image tensor

    vector_field:
        [B, 2, H, W]
        Channel 0: horizontal displacement in pixels
        Channel 1: vertical displacement in pixels

    Returns:
        Warped image [B, C, H, W]
    """
    if image.ndim != 4:
        raise ValueError("image must have shape [B, C, H, W]")

    batch, channels, height, width = image.shape

    if vector_field.shape != (batch, 2, height, width):
        raise ValueError(
            "vector_field must have shape [B, 2, H, W]"
        )

    # Base pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=image.dtype),
        torch.arange(width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )

    x = x.unsqueeze(0).expand(batch, -1, -1)
    y = y.unsqueeze(0).expand(batch, -1, -1)

    # Convert pixel displacements into normalized grid coordinates.
    dx = vector_field[:, 0]
    dy = vector_field[:, 1]

    if width > 1:
        x = x + dx
        x = 2.0 * x / (width - 1) - 1.0
    else:
        x = torch.zeros_like(x)

    if height > 1:
        y = y + dy
        y = 2.0 * y / (height - 1) - 1.0
    else:
        y = torch.zeros_like(y)

    sampling_grid = torch.stack([x, y], dim=-1)

    return F.grid_sample(
        image,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def adjust_brightness(warped_image, brightness_map):
    """
    Apply a predicted brightness map to a warped image.

    brightness_map must be in [0, 1].
    The map is interpreted as a multiplicative brightness mask.
    """
    if warped_image.ndim != 4:
        raise ValueError("warped_image must have shape [B, C, H, W]")

    batch, _, height, width = warped_image.shape

    if brightness_map.shape != (batch, 1, height, width):
        raise ValueError(
            "brightness_map must have shape [B, 1, H, W]"
        )

    return warped_image * brightness_map


def correct_eye_image(image, vector_field, brightness_map):
    """Warp the image, then apply the brightness adjustment."""
    warped = warp_image(image, vector_field)
    corrected = adjust_brightness(warped, brightness_map)
    return corrected


if __name__ == "__main__":
    image = torch.rand(2, 3, 64, 32)
    vector_field = torch.zeros(2, 2, 64, 32)
    brightness_map = torch.ones(2, 1, 64, 32)

    result = correct_eye_image(
        image, vector_field, brightness_map
    )

    print("Input:", image.shape)
    print("Corrected:", result.shape)
    print("Maximum difference:", (result - image).abs().max().item())
