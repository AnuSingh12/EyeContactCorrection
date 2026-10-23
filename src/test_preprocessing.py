import torch
from src.preprocessing import (
    preprocess_eye_images,
    prepare_target_angles,
)

images = torch.rand(2, 1, 36, 60)
angles = torch.zeros(2, 2)

processed = preprocess_eye_images(images)
targets = prepare_target_angles(angles)

print("Input:", images.shape)
print("Processed:", processed.shape)
print("Target angles:", targets.shape)

assert processed.shape == (2, 3, 32, 64)
assert targets.shape == (2, 2)

# Replicated channels should be identical.
assert torch.allclose(processed[:, 0], processed[:, 1])
assert torch.allclose(processed[:, 1], processed[:, 2])

print("Preprocessing test passed.")