import idx2numpy
import torch
import numpy as np
import matplotlib.pyplot as plt

SIZE = 100

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

all_images = idx2numpy.convert_from_file("data/FashionMNIST/raw/t10k-images-idx3-ubyte")
all_images = all_images / 255.0

all_labels = idx2numpy.convert_from_file("data/FashionMNIST/raw/t10k-labels-idx1-ubyte")

# choose arg.size random images
assert len(all_images) == len(all_labels)
idx = np.random.choice(len(all_images), size=SIZE, replace=False)

print(f"indices={idx}")

# save images and labels
idx2numpy.convert_to_file(f"fashion-images-{SIZE}.idx", all_images[idx])
idx2numpy.convert_to_file(f"fashion-labels-{SIZE}.idx", all_labels[idx].astype(np.uint8))

# check if saving worked:
images = idx2numpy.convert_from_file(f"fashion-images-{SIZE}.idx")
labels = idx2numpy.convert_from_file(f"fashion-labels-{SIZE}.idx")

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

print(f"first image={images[0]}")

plt.imshow(images[0], cmap="gray")
plt.title(f"Labels: {labels[0].tolist()}")
plt.show()