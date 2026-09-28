#simple function to save a checkpoint as the dataset in case dataset generation has been stopped and there is only a checkpoint
from pathlib import Path
import torch

from dataset import ChessDataset
from config import DATASET_PATH, DATASET_CHECKPOINT_PATH

DATASET_CHECKPOINT_PATH = Path(DATASET_CHECKPOINT_PATH)
DATASET_PATH = Path(DATASET_PATH)

print(
        f"Loading checkpoint: {DATASET_CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    DATASET_CHECKPOINT_PATH,
    map_location="cpu"
)
boards = list(checkpoint["boards"])
targets = list(checkpoint["targets"])

# Convert the lists into tensors
boards = torch.stack(boards)
targets = torch.stack(targets)

# Save the finished dataset
torch.save(
    {
        "boards": boards,
        "targets": targets
    },
    DATASET_PATH
)


print()
print("Dataset saved!")
print(f"Boards:  {boards.shape}")
print(f"Targets: {targets.shape}")
print(f"Path:    {DATASET_PATH}")