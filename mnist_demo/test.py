from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix

from load_model import load_model

BATCH_SIZE = 128

model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
DATA_DIR = r"C:\Users\Gigabyte\Documents\GitHub\Neural_Networks\mnist_demo\data"

train_data = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model.eval()

predictions = []
targets = []

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)

        logits = model(x_test)
        predicted = logits.argmax(dim=1)

        predictions.extend(predicted.cpu().tolist())
        targets.extend(y_test.tolist())

print("Prediction distribution:")
print(Counter(predictions))

accuracy = accuracy_score(targets, predictions)
confusion = confusion_matrix(targets, predictions)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion)
