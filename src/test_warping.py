
import torch

from src.warping import warp_image, apply_brightness_map, correct_eye


def main():
    image = torch.rand(2, 1, 32, 64)
    zero_flow = torch.zeros(2, 2, 32, 64)
    brightness = torch.zeros(2, 1, 32, 64)

    warped = warp_image(image, zero_flow)
    corrected = correct_eye(image, zero_flow, brightness)

    print("Original:", image.shape)
    print("Warped:", warped.shape)
    print("Corrected:", corrected.shape)

    # Zero displacement and unit brightness should preserve the image.
    assert torch.allclose(warped, image, atol=1e-5)
    assert torch.allclose(corrected, image, atol=1e-5)

    print("Warping test passed.")


if __name__ == "__main__":
    main()
