
import os
import copy
import random
import urllib.request
import zipfile

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image


# =========================================================
# 1. Reproducibility and device
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# =========================================================
# 2. Project paths and automatic dataset download
# =========================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_DIR, "hymenoptera_data")
ZIP_PATH = os.path.join(PROJECT_DIR, "hymenoptera_data.zip")

DATA_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

if not os.path.isdir(DATA_DIR):
    print("Downloading the ants and bees dataset...")

    if not os.path.isfile(ZIP_PATH):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(PROJECT_DIR)

    print("Dataset downloaded and extracted.")

else:
    print("Dataset already exists.")


# =========================================================
# 3. Image transformations
# =========================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


# =========================================================
# 4. Load datasets
# =========================================================

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=val_transform
)

class_names = train_dataset.classes

print(f"Classes: {class_names}")
print(f"Class mapping: {train_dataset.class_to_idx}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Training and validation class mappings do not match!")


# =========================================================
# 5. Create DataLoaders
# =========================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# 6. Load pretrained ResNet18
# =========================================================

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze all pretrained parameters.
for param in model.parameters():
    param.requires_grad = False

# Replace the original 1000-class classifier with a 2-class classifier.
input_features = model.fc.in_features

model.fc = nn.Linear(input_features, len(class_names))

model = model.to(device)

print("\nResNet18 model loaded.")
print(f"New classifier: {input_features} -> {len(class_names)}")


# =========================================================
# 7. Loss function and optimizer
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


# =========================================================
# 8. Training and validation
# =========================================================

EPOCHS = 5

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

best_val_acc = -1.0
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for phase in ["train", "val"]:

        if phase == "train":
            model.train()
            data_loader = train_loader
        else:
            model.eval()
            data_loader = val_loader

        running_loss = 0.0
        running_correct = 0
        total_images = 0

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            if phase == "train":
                optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):

                outputs = model(images)

                loss = criterion(outputs, labels)

                predictions = torch.argmax(outputs, dim=1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            running_correct += (predictions == labels).sum().item()
            total_images += batch_size

        epoch_loss = running_loss / total_images
        epoch_acc = running_correct / total_images

        history[f"{phase}_loss"].append(epoch_loss)
        history[f"{phase}_acc"].append(epoch_acc)

        print(
            f"{phase.capitalize()} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_acc * 100:.2f}%"
        )

        # Save the best model using validation accuracy.
        if phase == "val" and epoch_acc > best_val_acc:

            best_val_acc = epoch_acc

            best_model_weights = copy.deepcopy(model.state_dict())

            checkpoint_path = os.path.join(
                PROJECT_DIR,
                "best_resnet18_ants_bees.pth"
            )

            torch.save(
                {
                    "model_state_dict": best_model_weights,
                    "class_names": class_names
                },
                checkpoint_path
            )

            print("Best model checkpoint saved.")


# =========================================================
# 9. Restore the best model
# =========================================================

model.load_state_dict(best_model_weights)
model.eval()

print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%")


# =========================================================
# 10. Plot loss curves
# =========================================================

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
plt.plot(epochs_range, history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transfer Learning - Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "transfer_learning_loss.png")
)
plt.close()


# =========================================================
# 11. Plot accuracy curves
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], marker="o", label="Train Accuracy")
plt.plot(epochs_range, history["val_acc"], marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transfer Learning - Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "transfer_learning_accuracy.png")
)
plt.close()


# =========================================================
# 12. Confusion matrix on validation data
# =========================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", values_format="d")
plt.title("Transfer Learning - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "transfer_learning_confusion_matrix.png")
)
plt.close()


# =========================================================
# 13. Show sample predictions
# =========================================================

sample_count = min(6, len(val_dataset))

sample_indices = random.sample(
    range(len(val_dataset)),
    sample_count
)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

axes = axes.flatten()

for plot_index, sample_index in enumerate(sample_indices):

    image_tensor, true_label = val_dataset[sample_index]

    image_path, _ = val_dataset.samples[sample_index]

    original_image = Image.open(image_path).convert("RGB")

    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    is_correct = predicted_label == true_label

    axes[plot_index].imshow(original_image)
    axes[plot_index].set_title(
        f"True: {class_names[true_label]}\n"
        f"Predicted: {class_names[predicted_label]}\n"
        f"{'Correct' if is_correct else 'Incorrect'}"
    )
    axes[plot_index].axis("off")

for unused_index in range(sample_count, len(axes)):
    axes[unused_index].axis("off")

plt.suptitle("ResNet18 - Sample Predictions")
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "transfer_learning_predictions.png")
)
plt.close()


# =========================================================
# 14. Final report
# =========================================================

print("\n" + "=" * 50)
print("TRANSFER LEARNING PROJECT COMPLETED")
print("=" * 50)

print(f"Classes: {class_names}")
print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
print("Saved files:")
print("- best_resnet18_ants_bees.pth")
print("- transfer_learning_loss.png")
print("- transfer_learning_accuracy.png")
print("- transfer_learning_confusion_matrix.png")
print("- transfer_learning_predictions.png")
