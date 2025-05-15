import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader

# --- Custom Dataset ---
class HandSignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Improved MLP Model ---
class AdvancedMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # --- Load Data ---
    X, y = [], []
    data_dir = "data"

    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.endswith(".npy"):
                file_path = os.path.join(label_path, file)
                raw = np.load(file_path)
                # Nếu bạn muốn chuẩn hóa lại lúc này, uncomment:
                # raw = normalize_landmarks(raw)
                X.append(raw)
                y.append(label_folder)

    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    np.save("classes.npy", le.classes_)

    # --- Split Data ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_dataset = HandSignDataset(X_train, y_train)
    val_dataset = HandSignDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # --- Model Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedMLP(input_size=63, num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(40):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * y_batch.size(0)
            correct_train += (outputs.argmax(1) == y_batch).sum().item()
            total_train += y_batch.size(0)

        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item() * y_batch.size(0)
                correct_val += (outputs.argmax(1) == y_batch).sum().item()
                total_val += y_batch.size(0)

        avg_train_loss = total_train_loss / total_train
        avg_val_loss = total_val_loss / total_val
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/40 - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")

    # --- Save Model ---
    torch.save(model.state_dict(), "model.pth")

    # --- Plot Loss ---
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 4))
    plt.plot(train_accuracies, label='Train Accuracy', color='green')
    plt.plot(val_accuracies, label='Val Accuracy', color='red')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix ---
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
