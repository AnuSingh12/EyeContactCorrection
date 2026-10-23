
import numpy as np
from pathlib import Path
from collections import defaultdict

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "synthetic"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "processed"

# Load dataset metadata
names = np.load(DATASET_DIR / "dataset_names.npy")
gaze = np.load(DATASET_DIR / "dataset_y_gaze_angles.npy")

print("--- Dataset Verification ---")
print("Total samples:", len(names))
print("Gaze shape:", gaze.shape)

# Convert gaze labels from radians to degrees
gaze_degrees = np.degrees(gaze)

# Group samples by subject/configuration ID and head pose
groups = defaultdict(list)

for i, name in enumerate(names):
    parts = str(name).strip("_").split("_")

    if len(parts) != 5:
        continue

    key = (parts[0], parts[1], parts[2])
    groups[key].append(i)

# Find center and non-center gaze pairs
pairs = []
groups_with_pairs = 0
groups_without_center = 0

for key, indices in groups.items():
    center_indices = [
        i for i in indices
        if np.allclose(gaze_degrees[i], [0, 0], atol=1e-5)
    ]

    noncenter_indices = [
        i for i in indices
        if not np.allclose(gaze_degrees[i], [0, 0], atol=1e-5)
    ]

    if not center_indices:
        groups_without_center += 1
        continue

    if not noncenter_indices:
        continue

    # Expect one center sample for each ID + head pose.
    if len(center_indices) != 1:
        print(
            "Warning: expected one center sample for group",
            key,
            "| Found:",
            len(center_indices)
        )
        continue

    center_idx = center_indices[0]

    for input_idx in noncenter_indices:
        pairs.append((input_idx, center_idx))

    groups_with_pairs += 1

# Save index pairs without copying image data
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
output_path = OUTPUT_DIR / "center_pairs.npy"
np.save(output_path, pairs)

print("\n--- Pair Generation Results ---")
print("Total groups:", len(groups))
print("Groups with valid pairs:", groups_with_pairs)
print("Groups without center gaze:", groups_without_center)
print("Total input-target pairs:", len(pairs))
print("Saved indices to:", output_path)

if len(pairs):
    input_idx, target_idx = pairs[0]

    print("\n--- First Pair Preview ---")
    print("Input index:", input_idx)
    print("Input filename:", names[input_idx])
    print("Input gaze (degrees):",
          np.round(gaze_degrees[input_idx], 2))

    print("Target index:", target_idx)
    print("Target filename:", names[target_idx])
    print("Target gaze (degrees):",
          np.round(gaze_degrees[target_idx], 2))

    # Verify that both samples share the same group.
    input_parts = str(names[input_idx]).strip("_").split("_")
    target_parts = str(names[target_idx]).strip("_").split("_")

    print(
        "Same ID and head pose:",
        input_parts[:3] == target_parts[:3]
    )

print("\nPair generation complete.")
