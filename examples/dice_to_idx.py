import idx2numpy
import torch
import numpy as np

from examples.datasets import create_dice_datasets

SIZE = 68

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

_, test_loader, _, _, _ = create_dice_datasets(SIZE, normalise=False, seed=SEED)


def to_indices(label_vec):
    return np.where(label_vec == 1)[0]


all_images, all_labels = [], []

for imgs, labels in test_loader:
    imgs = imgs.cpu()  # imgs: [B, C, H, W]
    labels = labels.cpu()

    imgs_np = imgs.numpy()
    labels_np = labels.numpy()

    all_images.append(imgs_np)
    all_labels.append(labels_np)

all_images = np.concatenate(all_images, axis=0)  # [N, C, H, W]
all_labels = np.concatenate(all_labels, axis=0)  # one-hot e.g. [1,0,1,0,1]
all_labels = np.array([to_indices(label) for label in all_labels])  # e.g. [1,3,5]

print(all_images.dtype)
print(all_images.min(), all_images.max())

N = len(all_images)

# shuffle once
perm = np.random.permutation(N)

chunks = [perm[i : i + SIZE] for i in range(0, N, SIZE)]

print(f"total images: {N} number of chunks: {len(chunks)}")

# just a check
flat = np.concatenate(chunks)

assert len(flat) == len(set(flat)), "duplicate indices!"  # nosec
assert sorted(flat.tolist()) == list(range(N)), "missing indices!"  # nosec

for chunk_id, idx in enumerate(chunks):
    imgs = all_images[idx]
    labels = all_labels[idx]

    print(f"chunk {chunk_id}: size={len(idx)}, indices={idx}")

    idx2numpy.convert_to_file(f"dice-images-size{SIZE}-chunk{chunk_id}.idx", imgs)
    idx2numpy.convert_to_file(
        f"dice-labels-size{SIZE}-chunk{chunk_id}.idx", labels.astype(np.uint8)
    )

# check if saving worked:
images = idx2numpy.convert_from_file(f"dice-images-size{SIZE}-chunk0.idx")
labels = idx2numpy.convert_from_file(f"dice-labels-size{SIZE}-chunk0.idx")

print(images.dtype)
print(images.min(), images.max())

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

print(f"first image={images[0]}")

print(f"first label={labels[0]}")

# plt.imshow(np.transpose(images[0], (1, 2, 0)))
# plt.title(f"Labels: {labels[0].tolist()}")
# plt.show()
