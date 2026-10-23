
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.ecc_net import ECCNet
from src.gaze_angles import angles_to_cartesian
from src.warping import warp_image, apply_brightness_map
from src.losses import (
    correction_loss,
    reconstruction_loss,
    total_loss,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "synthetic"
PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

IMAGE_FILE = DATASET_DIR / "dataset_x_img.npy"
GAZE_FILE = DATASET_DIR / "dataset_y_gaze_angles.npy"
PAIR_FILE = PROCESSED_DIR / "center_pairs.npy"

CHECKPOINT_FILE = MODEL_DIR / "eccnet_latest.pth"
BEST_MODEL_FILE = MODEL_DIR / "eccnet_best.pth"

SEED = 42
BATCH_SIZE = 8
EPOCHS = 10                 # Total epochs, including resumed epochs
LEARNING_RATE = 1e-4
VALIDATION_RATIO = 0.10
NUM_WORKERS = 0

DEVICE = torch.device("cpu")


# ==================================================
# 2. REPRODUCIBILITY
# ==================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==================================================
# 3. DATASET
# ==================================================

class ECCPairDataset(Dataset):
    def __init__(self, images, gaze_angles, pairs):
        self.images = images
        self.gaze_angles = gaze_angles
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_idx, target_idx = map(int, self.pairs[index])

        input_image = np.array(
            self.images[input_idx], dtype=np.float32, copy=True
        )
        target_image = np.array(
            self.images[target_idx], dtype=np.float32, copy=True
        )
        input_angle = np.array(
            self.gaze_angles[input_idx], dtype=np.float32, copy=True
        )

        return {
            "input_image": torch.from_numpy(input_image).permute(2, 0, 1),
            "target_image": torch.from_numpy(target_image).permute(2, 0, 1),
            "input_angle": torch.from_numpy(input_angle),
        }


# ==================================================
# 4. IMAGE PREPROCESSING
# ==================================================

def prepare_images(images):
    """
    Input:  [B, 1, 36, 60]
    Output: [B, 3, 32, 64]
    """

    images = images.float()

    images = F.interpolate(
        images,
        size=(32, 64),
        mode="bilinear",
        align_corners=False,
    )

    return images.repeat(1, 3, 1, 1)


# ==================================================
# 5. WARPING AND BRIGHTNESS
# ==================================================

def transform_image(model, image, requested_angle):
    vector_field, brightness_map = model(image, requested_angle)
    warped = warp_image(image, vector_field)
    corrected = apply_brightness_map(warped, brightness_map)
    return corrected


# ==================================================
# 6. ONE TRAINING / VALIDATION EPOCH
# ==================================================

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_sum = 0.0
    correction_sum = 0.0
    reconstruction_sum = 0.0
    sample_count = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            original = prepare_images(
                batch["input_image"].to(DEVICE)
            )
            center_target = prepare_images(
                batch["target_image"].to(DEVICE)
            )
            input_angle = batch["input_angle"].to(DEVICE)

            # Provisional angle-to-Cartesian convention.
            input_cartesian = angles_to_cartesian(input_angle)

            # Forward: non-center gaze -> center gaze.
            center_request = torch.zeros_like(input_cartesian)

            corrected = transform_image(
                model, original, center_request
            )
            lc = correction_loss(corrected, center_target)

            # Reverse: center gaze -> original gaze.
            reconstructed = transform_image(
                model, center_target, input_cartesian
            )
            lr = reconstruction_loss(reconstructed, original)

            loss = total_loss(lc, lr)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

            batch_size = original.shape[0]
            sample_count += batch_size

            total_sum += loss.item() * batch_size
            correction_sum += lc.item() * batch_size
            reconstruction_sum += lr.item() * batch_size

    if sample_count == 0:
        raise RuntimeError("DataLoader returned no samples.")

    return {
        "total": total_sum / sample_count,
        "correction": correction_sum / sample_count,
        "reconstruction": reconstruction_sum / sample_count,
    }


# ==================================================
# 7. CHECKPOINTS
# ==================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    best_validation_loss,
    train_generator,
    config,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_validation_loss,
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "config": config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    train_generator,
    config,
):
    # Checkpoints created by this script contain tensors and
    # basic Python values; no custom objects are required.
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )

    saved_config = checkpoint.get("config", {})

    for key in ("seed", "batch_size", "learning_rate",
                "validation_ratio"):
        if saved_config.get(key) != config[key]:
            raise ValueError(
                f"Checkpoint configuration mismatch for '{key}'. "
                f"Saved={saved_config.get(key)}, current={config[key]}. "
                "Use the matching configuration or deliberately start "
                "a fresh run with a separate checkpoint."
            )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "train_generator_state" in checkpoint:
        train_generator.set_state(checkpoint["train_generator_state"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_loss = float(
        checkpoint.get("best_validation_loss", float("inf"))
    )

    return start_epoch, best_loss


# ==================================================
# 8. MAIN TRAINING PIPELINE
# ==================================================

def main():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for path in (IMAGE_FILE, GAZE_FILE, PAIR_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading dataset metadata...")

    # Memory-map the large image array instead of loading all images
    # into RAM at once.
    images = np.load(IMAGE_FILE, mmap_mode="r")
    gaze_angles = np.load(GAZE_FILE, mmap_mode="r")
    pairs = np.load(PAIR_FILE)

    if images.ndim != 4 or images.shape[-1] != 1:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    if gaze_angles.shape != (len(images), 2):
        raise ValueError(
            f"Unexpected gaze-angle shape: {gaze_angles.shape}"
        )

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"Unexpected pair shape: {pairs.shape}")

    if len(pairs) == 0:
        raise ValueError("No training pairs found.")

    if pairs.min() < 0 or pairs.max() >= len(images):
        raise ValueError("Pair indices fall outside the dataset.")

    # Group-aware split: all pairs sharing a center target index
    # remain together in either training or validation.
    group_ids = np.unique(pairs[:, 1])

    rng = np.random.default_rng(SEED)
    rng.shuffle(group_ids)

    validation_count = max(
        1, int(len(group_ids) * VALIDATION_RATIO)
    )

    validation_groups = group_ids[:validation_count]
    training_groups = group_ids[validation_count:]

    train_pairs = pairs[
        np.isin(pairs[:, 1], training_groups)
    ]
    validation_pairs = pairs[
        np.isin(pairs[:, 1], validation_groups)
    ]

    if len(train_pairs) == 0 or len(validation_pairs) == 0:
        raise RuntimeError("Training or validation split is empty.")

    train_dataset = ECCPairDataset(images, gaze_angles, train_pairs)
    validation_dataset = ECCPairDataset(
        images, gaze_angles, validation_pairs
    )

    # Dedicated generator lets us save and restore shuffle state.
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Device: {DEVICE}")
    print(f"Total images: {len(images)}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Training pairs: {len(train_dataset)}")
    print(f"Validation pairs: {len(validation_dataset)}")
    print(f"Training groups: {len(training_groups)}")
    print(f"Validation groups: {len(validation_groups)}")
    print(f"Total epochs configured: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    model = ECCNet().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    config = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "validation_ratio": VALIDATION_RATIO,
    }

    start_epoch = 1
    best_validation_loss = float("inf")

    if CHECKPOINT_FILE.exists():
        print("\nLatest checkpoint found. Loading it...")

        start_epoch, best_validation_loss = load_checkpoint(
            CHECKPOINT_FILE,
            model,
            optimizer,
            train_generator,
            config,
        )

        print(f"Checkpoint loaded. Next epoch: {start_epoch}")

    if start_epoch > EPOCHS:
        print(
            f"\nTraining already reached epoch {start_epoch - 1}. "
            f"Configured total is {EPOCHS}; nothing to run."
        )
        print("To intentionally start a fresh run, first choose "
              "different checkpoint filenames.")
        return

    # EPOCHS is the final total epoch, not extra epochs per resume.
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model, train_loader, optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader
        )

        elapsed = time.perf_counter() - epoch_start

        print(f"\nEpoch {epoch}/{EPOCHS} ({elapsed / 60:.1f} min)")
        print(
            "Train: "
            f"total={train_metrics['total']:.6f}, "
            f"correction={train_metrics['correction']:.6f}, "
            f"reconstruction={train_metrics['reconstruction']:.6f}"
        )
        print(
            "Validation: "
            f"total={validation_metrics['total']:.6f}, "
            f"correction={validation_metrics['correction']:.6f}, "
            f"reconstruction={validation_metrics['reconstruction']:.6f}"
        )

        if not np.isfinite(validation_metrics["total"]):
            raise RuntimeError(
                "Validation loss is not finite. "
                "Training stopped to avoid saving an invalid best model."
            )

        if validation_metrics["total"] < best_validation_loss:
            best_validation_loss = validation_metrics["total"]

            save_checkpoint(
                BEST_MODEL_FILE,
                epoch,
                model,
                optimizer,
                best_validation_loss,
                train_generator,
                config,
            )
            print("Best model saved.")

        # Save the latest checkpoint at the end of every epoch.
        save_checkpoint(
            CHECKPOINT_FILE,
            epoch,
            model,
            optimizer,
            best_validation_loss,
            train_generator,
            config,
        )
        print("Latest checkpoint saved.")

    print("\nTraining finished.")
    print(f"Latest checkpoint: {CHECKPOINT_FILE}")
    print(f"Best model: {BEST_MODEL_FILE}")


if __name__ == "__main__":
    main()
