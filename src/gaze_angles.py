
import torch


def angles_to_cartesian(angles):
    """
    Convert gaze angles [pitch, yaw] in radians
    into horizontal and vertical components.

    Convention used:
        horizontal = cos(pitch) * sin(yaw)
        vertical   = sin(pitch)

    Note:
        This is a conventional projection, not a formula
        explicitly specified in the ECC-Net paper.
    """
    if angles.ndim != 2 or angles.shape[1] != 2:
        raise ValueError(
            f"Expected angles with shape [B, 2], got {angles.shape}"
        )

    angles = angles.float()
    pitch = angles[:, 0]
    yaw = angles[:, 1]

    horizontal = torch.cos(pitch) * torch.sin(yaw)
    vertical = torch.sin(pitch)

    return torch.stack(
        [horizontal, vertical], dim=1
    )


if __name__ == "__main__":
    test_angles = torch.tensor([
        [0.0, 0.0],
        [-0.17453292, -0.17453292],
        [-0.17453292, 0.0],
    ])

    result = angles_to_cartesian(test_angles)

    print("Input angles (radians):")
    print(test_angles)

    print("\nCartesian components:")
    print(result)

    assert result.shape == (3, 2)
    assert torch.allclose(
        result[0], torch.zeros(2), atol=1e-6
    )
    assert torch.isfinite(result).all()

    print("\nGaze-angle conversion test passed.")
