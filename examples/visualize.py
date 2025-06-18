import matplotlib.pyplot as plt
from cifar_rgb.datasets import create_cifar_rgb_loaders
from cifar_rgb.transforms import grayscale

train_loader, _ = create_cifar_rgb_loaders(
    batch_size=4, transform=grayscale, train_ratio=0.8, random_seed=42
)

train_imgs, train_labels = next(iter(train_loader))
img_rgb = train_imgs[0]
label = train_labels[0].item()

r_channel = img_rgb[0]  # Red channel
g_channel = img_rgb[1]  # Green channel
b_channel = img_rgb[2]  # Blue channel

plt.figure(figsize=(10, 3))

# RGB composite
plt.subplot(1, 4, 1)
plt.imshow(img_rgb.permute(1, 2, 0).numpy())
plt.title(f"RGB (Label: {label})")
plt.axis("off")

# Red (mode1)
plt.subplot(1, 4, 2)
plt.imshow(r_channel.numpy(), cmap="gray")
plt.title("R / Mode 1")
plt.axis("off")

# Green (mode2)
plt.subplot(1, 4, 3)
plt.imshow(g_channel.numpy(), cmap="gray")
plt.title("G / Mode 2")
plt.axis("off")

# Blue (mode3)
plt.subplot(1, 4, 4)
plt.imshow(b_channel.numpy(), cmap="gray")
plt.title("B / Mode 3")
plt.axis("off")

plt.tight_layout()
plt.show()
