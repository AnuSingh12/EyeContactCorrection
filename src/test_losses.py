
import torch

from src.losses import (
    correction_loss,
    reconstruction_loss,
    total_loss,
)


def main():
    original = torch.rand(2, 1, 32, 64)
    target = torch.rand(2, 1, 32, 64)

    # Dummy images for testing only; these are not model predictions.
    corrected = torch.rand(2, 1, 32, 64, requires_grad=True)
    reconstructed = torch.rand(2, 1, 32, 64, requires_grad=True)

    lc = correction_loss(corrected, target)
    lr = reconstruction_loss(reconstructed, original)
    loss = total_loss(lc, lr)

    print("Correction loss:", lc.item())
    print("Reconstruction loss:", lr.item())
    print("Total loss:", loss.item())

    loss.backward()

    assert torch.isfinite(loss)
    assert corrected.grad is not None
    assert reconstructed.grad is not None

    print("Loss test passed.")
    print("Gradient test passed.")


if __name__ == "__main__":
    main()
