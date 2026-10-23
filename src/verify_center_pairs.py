
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset" / "synthetic"
PROCESSED_DIR = ROOT / "dataset" / "processed"

pairs = np.load(
    PROCESSED_DIR / "center_pairs.npy",
    mmap_mode="r",
)

gaze_angles = np.load(
    DATASET_DIR / "dataset_y_gaze_angles.npy",
    mmap_mode="r",
)

groups = np.load(
    PROCESSED_DIR / "gaze_training_groups.npy",
    mmap_mode="r",
)

print("Total pairs:", len(pairs))

# Check index ranges.
assert pairs.ndim == 2 and pairs.shape[1] == 2
assert pairs.min() >= 0
assert pairs.max() < len(gaze_angles)

input_indices = pairs[:, 0]
target_indices = pairs[:, 1]

# Input and target must be different samples.
assert np.all(input_indices != target_indices)

# Every target should have center gaze: [0, 0].
target_angles = gaze_angles[target_indices]
center_mask = np.all(
    np.isclose(target_angles, 0.0, atol=1e-6),
    axis=1,
)

print("Center-gaze targets:", int(center_mask.sum()))
print("Non-center targets:", int((~center_mask).sum()))

# Map every dataset index to its group.
group_of = np.full(len(gaze_angles), -1, dtype=np.int32)

for group_id, group in enumerate(groups):
    group_of[group] = group_id

# Input and target must belong to the same group.
same_group = (
    group_of[input_indices] == group_of[target_indices]
)

print("Same-group pairs:", int(same_group.sum()))
print("Different-group pairs:", int((~same_group).sum()))

# Show the first pair for manual inspection.
input_idx, target_idx = map(int, pairs[0])

print("\nFirst pair:")
print("Input index:", input_idx)
print("Input gaze (radians):", gaze_angles[input_idx])
print("Target index:", target_idx)
print("Target gaze (radians):", gaze_angles[target_idx])

assert center_mask.all(), "Some targets are not center gaze!"
assert same_group.all(), "Some pairs cross group boundaries!"

print("\nAll center-pair checks passed.")
