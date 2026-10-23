
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "synthetic"
PAIR_FILE = PROJECT_DIR / "dataset" / "processed" / "center_pairs.npy"

# Load pair indices and metadata
pairs = np.load(PAIR_FILE)
names = np.load(DATASET_DIR / "dataset_names.npy")
gaze = np.load(DATASET_DIR / "dataset_y_gaze_angles.npy")

# Load images without loading the entire dataset into RAM
images = np.load(
    DATASET_DIR / "dataset_x_img.npy",
    mmap_mode="r"
)

print("--- Image Inspection ---")
print("Image dataset shape:", images.shape)
print("Image data type:", images.dtype)
print("Total pairs:", len(pairs))

# Inspect the first generated pair
input_idx, target_idx = pairs[0]

input_image = images[input_idx, :, :, 0]
target_image = images[target_idx, :, :, 0]

input_gaze = np.round(np.degrees(gaze[input_idx]), 2)
target_gaze = np.round(np.degrees(gaze[target_idx]), 2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(input_image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title(
    f"Input gaze: {input_gaze}\n{names[input_idx]}"
)
axes[0].axis("off")

axes[1].imshow(target_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(
    f"Target gaze: {target_gaze}\n{names[target_idx]}"
)
axes[1].axis("off")

plt.tight_layout()
plt.show()

print("\n--- Pair Details ---")
print("Input index:", input_idx)
print("Target index:", target_idx)
print("Input image shape:", input_image.shape)
print("Target image shape:", target_image.shape)
print("Input gaze (degrees):", input_gaze)
print("Target gaze (degrees):", target_gaze)

input_parts = str(names[input_idx]).strip("_").split("_")
target_parts = str(names[target_idx]).strip("_").split("_")

print(
    "Same ID and head pose:",
    input_parts[:3] == target_parts[:3]
)
