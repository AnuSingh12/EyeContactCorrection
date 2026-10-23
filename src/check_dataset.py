
import numpy as np
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "synthetic"

files = [
    "dataset_x_img.npy",
    "dataset_y_gaze_angles.npy",
    "dataset_x_head_angles.npy",
    "dataset_names.npy",
]

for filename in files:
    path = DATASET_DIR / filename

    print(f"\n--- {filename} ---")

    if not path.exists():
        print("File not found!")
        continue

    data = np.load(path, allow_pickle=False)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Size:", f"{data.nbytes / (1024**2):.2f} MB")

    if np.issubdtype(data.dtype, np.number):
        print("First values:", data.reshape(-1)[:10])



import numpy as np
from pathlib import Path

data_dir = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "synthetic"
)

names = np.load(data_dir / "dataset_names.npy")

print("\nFirst 20 dataset names:")
for name in names[:20]:
    print(repr(name))

print("\nUnique names:", len(np.unique(names)))



print("\n--- Dataset name analysis ---")

for index in [0, 1, 100, 1000, 10000, len(names) - 1]:
    print(f"Index {index}: {names[index]}")

print("\nFirst 10 gaze angles (degrees):")
gaze = np.load(data_dir / "dataset_y_gaze_angles.npy")
print(np.degrees(gaze[:10]))

print("\nFirst 10 head angles (degrees):")
head = np.load(data_dir / "dataset_x_head_angles.npy")
print(np.degrees(head[:10]))



import matplotlib.pyplot as plt

images = np.load(
    DATASET_DIR / "dataset_x_img.npy",
    mmap_mode="r"
)
gaze = np.load(DATASET_DIR / "dataset_y_gaze_angles.npy")
head = np.load(DATASET_DIR / "dataset_x_head_angles.npy")
names = np.load(DATASET_DIR / "dataset_names.npy")

indices = [0, 1, 6, 12, 13, 100]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for ax, i in zip(axes.flat, indices):
    ax.imshow(images[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"{names[i]}\n"
        f"Gaze: {np.round(np.degrees(gaze[i])).astype(int)}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()



# Check whether images with different gaze angles
# share the same subject/head-pose configuration.

print("\n--- Matching configuration check ---")

for i in [0, 1, 6, 12, 13, 100]:
    print(
        f"Index: {i} | "
        f"Name: {names[i]} | "
        f"Gaze: {np.round(np.degrees(gaze[i])).astype(int)} | "
        f"Head: {np.round(np.degrees(head[i])).astype(int)}"
    )

print("\n--- Same prefix examples ---")
for i in range(13):
    print(names[i])



from collections import Counter

print("\n--- Dataset distribution ---")

# Count images for each head-angle pair
head_degrees = np.round(np.degrees(head)).astype(int)
head_pairs = [tuple(row) for row in head_degrees]

counts = Counter(head_pairs)

print("Unique head-angle pairs:", len(counts))
print("Most common head-angle pairs:")
for pair, count in counts.most_common(10):
    print(f"Head angles {pair}: {count} images")

# Count unique gaze directions
gaze_degrees = np.round(np.degrees(gaze)).astype(int)
gaze_pairs = [tuple(row) for row in gaze_degrees]

print("\nUnique gaze-angle pairs:", len(set(gaze_pairs)))

print("\n--- Dataset distribution ---")

head_degrees = np.round(np.degrees(head)).astype(int)
head_pairs = [tuple(row) for row in head_degrees]

from collections import Counter

counts = Counter(head_pairs)

print("Unique head-angle pairs:", len(counts))
print("Most common head-angle pairs:")
for pair, count in counts.most_common(10):
    print(f"Head angles {pair}: {count} images")

gaze_degrees = np.round(np.degrees(gaze)).astype(int)
gaze_pairs = [tuple(row) for row in gaze_degrees]

print("\nUnique gaze-angle pairs:", len(set(gaze_pairs)))


print("\n--- Matching gaze configurations ---")

# Inspect the first configuration group.
# The first number in the name appears to identify a subject/configuration;
# verify grouping from the dataset structure before relying on it.
for i in range(13):
    print(
        f"Name: {names[i]} | "
        f"Gaze: {np.round(np.degrees(gaze[i])).astype(int)} | "
        f"Head: {np.round(np.degrees(head[i])).astype(int)}"
    )

# Count samples with the same head pose and gaze direction.
combined = [
    (tuple(h), tuple(g))
    for h, g in zip(head_degrees, gaze_degrees)
]

print("Unique head-pose + gaze combinations:", len(set(combined)))
print("Total samples:", len(combined))
