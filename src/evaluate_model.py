
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.ecc_net import ECCNet
from src.gaze_angles import angles_to_cartesian
from src.warping import warp_image, apply_brightness_map


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "synthetic"
PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation"

IMAGE_FILE = DATASET_DIR / "dataset_x_img.npy"
GAZE_FILE = DATASET_DIR / "dataset_y_gaze_angles.npy"
PAIR_FILE = PROCESSED_DIR / "center_pairs.npy"
MODEL_FILE = MODEL_DIR / "eccnet_best.pth"

DEVICE = torch.device("cpu")
NUM_SAMPLES = 4
SEED = 42


# ==================================================
# 2. PREPROCESSING
# ==================================================

def prepare_images(images):
    """Convert [B,1,36,60] images to [B,3,32,64]."""
    images = images.float()

    images = F.interpolate(
        images,
        size=(32, 64),
        mode="bilinear",
        align_corners=False,
    )

    return images.repeat(1, 3, 1, 1)


# ==================================================
# 3. LOAD TRAINED MODEL
# ==================================================

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    model = ECCNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {MODEL_FILE}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    return model


# ==================================================
# 4. MODEL INFERENCE
# ==================================================

def correct_image(model, image, requested_angle):
    """Return warped image, corrected image and predicted maps."""
    vector_field, brightness_map = model(
        image,
        requested_angle,
    )

    warped = warp_image(image, vector_field)

    corrected = apply_brightness_map(
        warped,
        brightness_map,
    )

    return warped, corrected, vector_field, brightness_map


# ==================================================
# 5. EVALUATION
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (IMAGE_FILE, GAZE_FILE, PAIR_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    print("Loading dataset...")

    images = np.load(IMAGE_FILE, mmap_mode="r")
    gaze_angles = np.load(GAZE_FILE, mmap_mode="r")
    pairs = np.load(PAIR_FILE)

    if images.ndim != 4 or images.shape[-1] != 1:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    if gaze_angles.shape != (len(images), 2):
        raise ValueError(
            f"Unexpected gaze shape: {gaze_angles.shape}"
        )

    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError(f"Unexpected pair shape: {pairs.shape}")

    if pairs.min() < 0 or pairs.max() >= len(images):
        raise ValueError("Pair index outside dataset bounds.")

    model = load_model()

    rng = np.random.default_rng(SEED)
    sample_count = min(NUM_SAMPLES, len(pairs))

    selected_rows = rng.choice(
        len(pairs),
        size=sample_count,
        replace=False,
    )

    selected_pairs = pairs[selected_rows]

    titles = [
        "Original eye",
        "Center-gaze target",
        "Warped only",
        "Warped + brightness",
    ]

    figure, axes = plt.subplots(
        sample_count,
        4,
        figsize=(15, 3.5 * sample_count),
        squeeze=False,
    )

    original_mses = []
    warped_mses = []
    corrected_mses = []

    for row, (input_idx, target_idx) in enumerate(selected_pairs):
        input_idx = int(input_idx)
        target_idx = int(target_idx)

        input_array = np.array(
            images[input_idx],
            dtype=np.float32,
            copy=True,
        )

        target_array = np.array(
            images[target_idx],
            dtype=np.float32,
            copy=True,
        )

        original = torch.from_numpy(
            input_array
        ).permute(2, 0, 1).unsqueeze(0)

        target = torch.from_numpy(
            target_array
        ).permute(2, 0, 1).unsqueeze(0)

        original = prepare_images(original).to(DEVICE)
        target = prepare_images(target).to(DEVICE)

        input_angle = torch.from_numpy(
            np.array(
                gaze_angles[input_idx],
                dtype=np.float32,
                copy=True,
            )
        ).unsqueeze(0).to(DEVICE)

        # Request center gaze.
        center_request = torch.zeros_like(
            angles_to_cartesian(input_angle)
        )

        with torch.no_grad():
            warped, corrected, vector_field, brightness_map = (
                correct_image(
                    model,
                    original,
                    center_request,
                )
            )

        # Pixel-wise MSE against the center-gaze target.
        original_mse = F.mse_loss(
            original, target
        ).item()

        warped_mse = F.mse_loss(
            warped, target
        ).item()

        corrected_mse = F.mse_loss(
            corrected, target
        ).item()

        original_mses.append(original_mse)
        warped_mses.append(warped_mse)
        corrected_mses.append(corrected_mse)

        gaze_degrees = np.degrees(
            gaze_angles[input_idx]
        )

        print(f"\nSample {row + 1}")
        print(
            f"Input gaze (degrees): "
            f"{np.round(gaze_degrees, 2)}"
        )
        print(
            f"Vector field min/max: "
            f"{vector_field.min().item():.4f}, "
            f"{vector_field.max().item():.4f}"
        )
        print(
            f"Vector field mean abs: "
            f"{vector_field.abs().mean().item():.4f}"
        )
        print(
            f"Brightness min/max/mean: "
            f"{brightness_map.min().item():.4f}, "
            f"{brightness_map.max().item():.4f}, "
            f"{brightness_map.mean().item():.4f}"
        )
        print(f"Original vs target MSE: {original_mse:.6f}")
        print(f"Warped vs target MSE:   {warped_mse:.6f}")
        print(f"Corrected vs target MSE:{corrected_mse:.6f}")

        display_images = [
            original[0, 0].cpu().numpy(),
            target[0, 0].cpu().numpy(),
            warped[0, 0].cpu().numpy(),
            corrected[0, 0].cpu().numpy(),
        ]

        for col, display_image in enumerate(display_images):
            axes[row, col].imshow(
                np.clip(display_image, 0, 1),
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(titles[col])

        axes[row, 0].set_ylabel(
            f"Sample {row + 1}\n"
            f"Gaze: {gaze_degrees[0]:.1f}, "
            f"{gaze_degrees[1]:.1f} deg",
            fontsize=9,
        )

    figure.suptitle(
        "ECC-Net: Original, Target, Warped and Corrected",
        fontsize=14,
    )

    figure.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = (
        OUTPUT_DIR / "eccnet_full_diagnostic.png"
    )

    figure.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(figure)

    print("\n========== MEAN MSE SUMMARY ==========")
    print(f"Original vs target:  {np.mean(original_mses):.6f}")
    print(f"Warped vs target:    {np.mean(warped_mses):.6f}")
    print(f"Corrected vs target: {np.mean(corrected_mses):.6f}")

    print("\nEvaluation complete.")
    print(f"Saved image: {output_file}")


if __name__ == "__main__":
    main()
