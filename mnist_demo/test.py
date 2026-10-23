from collections import Counter

import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from load_model import load_model

start_time = time.time()

BATCH_SIZE = 128

print("Loading model...")
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model loaded. Device: {device}")
print(f"Time: {time.time() - start_time:.2f}s")

transform = transforms.ToTensor()
DATA_DIR = r"C:\Users\Gigabyte\Documents\GitHub\Neural_Networks\mnist_demo\data"

print("Loading test dataset...")
test_data = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)
print(f"Test dataset loaded: {len(test_data)} images")
print(f"Time: {time.time() - start_time:.2f}s")

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model.eval()

predictions = []
targets = []

print("Starting predictions...")

with torch.inference_mode():
    for batch_index, (x_test, y_test) in enumerate(test_loader):
        x_test = x_test.to(device)

        logits = model(x_test)
        predicted = logits.argmax(dim=1)

        predictions.extend(predicted.cpu().tolist())
        targets.extend(y_test.tolist())

        if (batch_index + 1) % 10 == 0:
            print(
                f"Processed {(batch_index + 1) * BATCH_SIZE} images "
                f"in {time.time() - start_time:.2f}s"
            )

print("Predictions finished.")
print(f"Time: {time.time() - start_time:.2f}s")

print("\nPrediction distribution:")
print(Counter(predictions))

accuracy = accuracy_score(targets, predictions)
confusion = confusion_matrix(targets, predictions)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion)
