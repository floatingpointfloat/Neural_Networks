from torchvision import datasets, transforms
from pathlib import Path

# dataset and dataloader - EMNIST balanced split
DATA_DIR = Path(__file__).resolve().parent / "data"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_data = datasets.EMNIST(
    root=DATA_DIR, split="balanced", train=True, download=True, transform=transform
)

test_data = datasets.EMNIST(
    root=DATA_DIR, split="balanced", train=False, download=True, transform=transform
)

print(train_data.classes)
