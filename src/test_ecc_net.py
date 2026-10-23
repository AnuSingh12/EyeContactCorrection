
import torch

from src.ecc_net import ECCNet


def main():
    torch.manual_seed(42)

    model = ECCNet()
    model.eval()

    # Match preprocessing.py output: [B, 3, 32, 64]
    image = torch.randn(2, 3, 32, 64)

    # Two target-angle components per image.
    target_angle = torch.zeros(2, 2)

    with torch.no_grad():
        vector_field, brightness_map = model(
            image, target_angle
        )

    print("Input image:", image.shape)
    print("Target angle:", target_angle.shape)
    print("Vector field:", vector_field.shape)
    print("Brightness map:", brightness_map.shape)

    assert vector_field.shape == (2, 2, 32, 64)
    assert brightness_map.shape == (2, 1, 32, 64)

    assert torch.isfinite(vector_field).all()
    assert torch.isfinite(brightness_map).all()

    assert brightness_map.min() >= 0
    assert brightness_map.max() <= 1

    print("ECC-Net model shape test passed.")


if __name__ == "__main__":
    main()
