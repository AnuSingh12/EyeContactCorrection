
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.gaze_angles import angles_to_cartesian


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset" / "synthetic"
PROCESSED_DIR = ROOT / "dataset" / "processed"


class ECCTrainingDataset(Dataset):
    """
    ECC-Net dataset using pre-generated center-gaze pairs.

    Images:
        Stored: [N, 36, 60, 1]
        Returned: [1, 36, 60]

    Angles:
        Raw angles: [2], in radians.
        Cartesian components: [2], using gaze_angles.py.

    Pairs:
        [input_index, target_index]
        Input gaze -> center-gaze target.
    """

    def __init__(self):
        self.images = np.load(
            DATASET_DIR / "dataset_x_img.npy",
            mmap_mode="r",
        )

        self.gaze_angles = np.load(
            DATASET_DIR / "dataset_y_gaze_angles.npy",
            mmap_mode="r",
        )

        self.pairs = np.load(
            PROCESSED_DIR / "center_pairs.npy",
            mmap_mode="r",
        )

        if self.images.ndim != 4 or self.images.shape[-1] != 1:
            raise ValueError(
                f"Expected images shaped [N, H, W, 1], "
                f"got {self.images.shape}"
            )

        if self.gaze_angles.shape != (len(self.images), 2):
            raise ValueError(
                f"Expected gaze angles shaped "
                f"({len(self.images)}, 2), "
                f"got {self.gaze_angles.shape}"
            )

        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError(
                f"Expected pairs shaped [N, 2], "
                f"got {self.pairs.shape}"
            )

        if len(self.pairs) == 0:
            raise ValueError("No training pairs found.")

        if (
            self.pairs.min() < 0
            or self.pairs.max() >= len(self.images)
        ):
            raise ValueError(
                "Pair indices are outside the dataset."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_index, target_index = map(
            int, self.pairs[index]
        )

        # Load the selected images.
        input_image = np.array(
            self.images[input_index],
            dtype=np.float32,
        )

        target_image = np.array(
            self.images[target_index],
            dtype=np.float32,
        )

        # Convert [H, W, 1] -> [1, H, W].
        input_image = torch.from_numpy(
            input_image
        ).permute(2, 0, 1)

        target_image = torch.from_numpy(
            target_image
        ).permute(2, 0, 1)

        # Read raw gaze angles in radians.
        input_angle = torch.from_numpy(
            np.array(
                self.gaze_angles[input_index],
                dtype=np.float32,
            ).copy()
        )

        target_angle = torch.from_numpy(
            np.array(
                self.gaze_angles[target_index],
                dtype=np.float32,
            ).copy()
        )

        # Convert angles into provisional Cartesian components.
        # See src/gaze_angles.py for the chosen convention.
        input_angle_cartesian = angles_to_cartesian(
            input_angle.unsqueeze(0)
        ).squeeze(0)

        target_angle_cartesian = angles_to_cartesian(
            target_angle.unsqueeze(0)
        ).squeeze(0)

        return {
            "input_image": input_image,
            "target_image": target_image,

            # Original angles.
            "input_angle": input_angle,
            "target_angle": target_angle,

            # Converted components.
            "input_angle_cartesian": input_angle_cartesian,
            "target_angle_cartesian": target_angle_cartesian,

            "input_index": input_index,
            "target_index": target_index,
        }


if __name__ == "__main__":
    dataset = ECCTrainingDataset()
    sample = dataset[0]

    print(f"Total training pairs: {len(dataset)}")

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(
                f"{key}: shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"value={value.tolist()}"
            )
        else:
            print(f"{key}: {value}")

    # Shape checks.
    assert sample["input_image"].shape == (1, 36, 60)
    assert sample["target_image"].shape == (1, 36, 60)

    assert sample["input_angle"].shape == (2,)
    assert sample["target_angle"].shape == (2,)

    assert sample["input_angle_cartesian"].shape == (2,)
    assert sample["target_angle_cartesian"].shape == (2,)

    assert sample["input_index"] != sample["target_index"]

    # The target of each verified center pair should be zero gaze.
    assert torch.allclose(
        sample["target_angle"],
        torch.zeros(2),
        atol=1e-5,
    )

    assert torch.allclose(
        sample["target_angle_cartesian"],
        torch.zeros(2),
        atol=1e-5,
    )

    print("\nTraining data sample test passed.")
