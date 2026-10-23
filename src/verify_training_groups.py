
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "synthetic"
PROCESSED = ROOT / "dataset" / "processed"

names = np.load(DATA / "dataset_names.npy", allow_pickle=True)
gaze = np.load(DATA / "dataset_y_gaze_angles.npy")
head = np.load(DATA / "dataset_x_head_angles.npy")
groups = np.load(PROCESSED / "gaze_training_groups.npy")

print("--- Group Verification ---")
print("Names:", len(names))
print("Gaze labels:", gaze.shape)
print("Head angles:", head.shape)
print("Groups:", groups.shape)

assert len(names) == len(gaze) == len(head)
assert groups.ndim == 2
assert groups.size == len(names)
assert groups.min() >= 0
assert groups.max() < len(names)
assert len(np.unique(groups)) == len(names)

bad_head_groups = 0
bad_name_groups = 0
gaze_counts = []
center_counts = []

for group in groups:
    # All samples in a group should have matching head angles.
    group_head = head[group]
    if not np.allclose(group_head, group_head[0], atol=1e-6):
        bad_head_groups += 1

    # Check that filenames agree on the grouping key.
    keys = [
        tuple(str(names[i]).strip().rstrip("_").split("_")[:3])
        for i in group
    ]
    if len(set(keys)) != 1:
        bad_name_groups += 1

    # Count distinct gaze directions.
    group_gaze = np.round(np.degrees(gaze[group]), 4)
    unique_gazes = np.unique(group_gaze, axis=0)
    gaze_counts.append(len(unique_gazes))

    # Count center-gaze samples.
    is_center = np.all(
        np.isclose(gaze[group], 0.0, atol=1e-5), axis=1
    )
    center_counts.append(int(is_center.sum()))

print("\n--- Results ---")
print("Groups with inconsistent head angles:", bad_head_groups)
print("Groups with inconsistent filename keys:", bad_name_groups)
print("Minimum unique gaze directions:", min(gaze_counts))
print("Maximum unique gaze directions:", max(gaze_counts))
print("Minimum center samples per group:", min(center_counts))
print("Maximum center samples per group:", max(center_counts))

print("\n--- First Group Sample ---")
for i in groups[0][:10]:
    print(
        "Index:", int(i),
        "| Name:", repr(names[i]),
        "| Gaze:", np.round(np.degrees(gaze[i]), 2),
        "| Head:", np.round(np.degrees(head[i]), 2),
    )

if bad_head_groups == 0 and bad_name_groups == 0:
    print("\nGrouping consistency checks passed.")
else:
    print("\nWARNING: Investigate inconsistent groups before training.")
