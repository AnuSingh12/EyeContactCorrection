
from pathlib import Path
from collections import defaultdict

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset" / "synthetic"
OUTPUT_DIR = ROOT / "dataset" / "processed"

NAMES_FILE = DATASET_DIR / "dataset_names.npy"
GAZE_FILE = DATASET_DIR / "dataset_y_gaze_angles.npy"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    names = np.load(NAMES_FILE, allow_pickle=True)
    gaze_angles = np.load(GAZE_FILE)

    if len(names) != len(gaze_angles):
        raise ValueError("Filename and gaze-angle counts do not match.")

    groups = defaultdict(list)

    for index, name in enumerate(names):
        filename = str(name).strip().rstrip("_")
        parts = filename.split("_")

        if len(parts) < 5:
            raise ValueError(
                f"Unexpected filename format at index {index}: {name}"
            )

        # Group by subject ID and both head-angle components.
        group_key = tuple(parts[:3])
        groups[group_key].append(index)

    group_keys = sorted(groups)
    group_sizes = {len(indices) for indices in groups.values()}

    if len(group_sizes) != 1:
        raise ValueError(
            f"Groups have different sizes: {sorted(group_sizes)}"
        )

    group_indices = np.array(
        [groups[key] for key in group_keys],
        dtype=np.int64,
    )

    output_file = OUTPUT_DIR / "gaze_training_groups.npy"
    np.save(output_file, group_indices)

    print("Groups:", len(group_indices))
    print("Images per group:", group_indices.shape[1])
    print("Total indexed images:", group_indices.size)
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()
