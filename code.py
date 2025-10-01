import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# ------------------------------------------
def flatten_to_tensor(row):
    # Take row values
    pixels = row[1:].values.astype(np.uint8)
    # Reshape into 28x28
    return pixels.reshape(28, 28)

# Create new dataframe
df_tensors = pd.DataFrame({
    "label": data.iloc[:, 0],  # first column as label
    "image": data.apply(flatten_to_tensor, axis=1)  # convert pixels -> 28x28 array
})
print(df_tensors.head())

# Example:
import matplotlib.pyplot as plt
plt.imshow(df_tensors.loc[3, "image"], cmap="gray")
plt.title(f"Label: {df_tensors.loc[3, 'label']}")
plt.show()
# -------------------------------------------
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class DigitDataset(Dataset):
    def __init__(self, df, transform=None):
        self.labels = df["label"].astype(int).values
        self.images = df["image"].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.asarray(img, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = np.transpose(img, (2, 0, 1))
            elif img.shape[0] == 1:
                pass
            else:
                try:
                    img = img.reshape(1, 28, 28)
                except Exception:
                    raise ValueError(f"Unexpected image shape {img.shape} at index {idx}")

        # Final check
        if img.shape != (1, 28, 28):
            raise ValueError(f"Image at index {idx} has wrong shape after processing: {img.shape}")

        img_tensor = torch.from_numpy(img).float()
        label = int(self.labels[idx])
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label
      # -------------------------------------------------------
      class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # -> 14x14

            # block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # -> 7x7

            # block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 7x7 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),                  # -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * 1 * 1, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)   # outputs raw logits
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x  # logits
# ------------------------------------------------------------
# Helper: train / eval
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)              # shape (B,10)
        loss = criterion(logits, labels)    # CrossEntropy expects logits
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
  # ----------------------------------------------------------
  def run_training(df_tensors,
                 batch_size=64,
                 lr=1e-3,
                 epochs=10,
                 val_split=0.1,
                 device=None,
                 save_path="best_mnist_cnn.pth"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_df, val_df = train_test_split(df_tensors, test_size=val_split,
                                        stratify=df_tensors["label"], random_state=SEED)

    train_ds = DigitDataset(train_df)
    val_ds = DigitDataset(val_df)

    # safer defaults for dataloader
    num_workers = 2
    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs}  "
              f"Train loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, save_path)

    print(f"Best val acc: {best_val_acc:.4f}")
    return model, history
    # --------------------------------------------------------
def load_model_from_checkpoint(model, ckpt_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    return model

def predict_with_probs(model, image_np):
    """
    image_np: numpy array either shape (28,28), (1,28,28), or (28,28,1).
    Returns: probs (10,) numpy array
    """
    model.eval()
    img = np.asarray(image_np, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # ensure (1,28,28)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim == 3:
        if img.shape[2] == 1:  # (28,28,1)
            img = np.transpose(img, (2,0,1))
        elif img.shape[0] != 1:
            try:
                img = img.reshape(1,28,28)
            except Exception:
                raise ValueError(f"Unexpected image shape: {img.shape}")

    if img.shape != (1,28,28):
        raise ValueError(f"After processing image shape is {img.shape}; expected (1,28,28)")

    tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1,1,28,28)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)                      # (1,10)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze(0)
    return probs
  # ----------------------------------------------------------
  # train
model, history = run_training(df_tensors, epochs=10, batch_size=64, lr=1e-3, save_path="best_mnist_cnn.pth")

# if trained
model = SimpleCNN(num_classes=10)
model = load_model_from_checkpoint(model, "best_mnist_cnn.pth")

# Predict on first row
probs = predict_with_probs(model, df_tensors.loc[0, "image"])
predicted_digit = int(np.argmax(probs))
print("pred:", predicted_digit)
print("top-5 probs:", list(zip(range(10), probs)))
