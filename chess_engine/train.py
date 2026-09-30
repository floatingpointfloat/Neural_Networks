import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from model import ChessValueNet
from config import DATASET_PATH

torch.backends.cudnn.benchmark = True #cuda optimization

ALLOW_TRAINING = True

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.0005

# Path to the checkpoint folder
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

#use CUDA if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ChessValueNet().to(device)

#master switch
if not ALLOW_TRAINING:
    raise RuntimeError("No Training currently allowed. Please do not train the model")

# Load the saved dataset
data = torch.load(
    DATASET_PATH,
    map_location="cpu"
)

boards = data["boards"]
targets = data["targets"]

print(f"Loaded dataset: {boards.shape[0]} positions")
print(f"Boards shape:   {boards.shape}")
print(f"Targets shape:  {targets.shape}")

dataset = TensorDataset(
    boards,
    targets
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0, #cuda optimization
    pin_memory=True #faster gpu vram usage
)

# Check the target distribution - only for debug
print()
print(f"Target mean: {targets.mean().item():.6f}")
print(f"Target std:  {targets.std().item():.6f}")
print(f"Target min:  {targets.min().item():.6f}")
print(f"Target max:  {targets.max().item():.6f}")
print()


#loss function
criterion = torch.nn.MSELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#best loss
best_loss = float("inf")

# Load the latest checkpoint if one exists
if LATEST_CHECKPOINT.exists():
    print(f"Loading checkpoint: {LATEST_CHECKPOINT}")
    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming from epoch {start_epoch}")
    # Use the saved loss as the current best loss
    if "loss" in checkpoint:
        best_loss = checkpoint["loss"]
else:
    print("No checkpoint found. Starting from scratch.")
    start_epoch = 0

#training loop
for epoch in range(start_epoch, EPOCHS + start_epoch):
    model.train()

    epoch_loss = 0.0
    batches = 0

    for boards, targets in loader:
        #move the data to the devide (gpu cuda/cpu)
        boards = boards.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        #reset the gradients to 0
        optimizer.zero_grad()

        #forward pass
        predictions = model(boards)

        # Remove the unnecessary dimension [64, 1] -> [64]
        predictions = predictions.squeeze(1)

        #loss calculation
        loss = criterion(predictions, targets)

        if batches % 1000 == 0:
            print(f"Batch {batches} | Loss: {loss.item():.6f}")

        #backpropagation aka pytorch does its magic
        loss.backward()

        #update the parameters
        optimizer.step()

        #track the loss
        epoch_loss += loss.item()
        batches += 1

    #average loss for this epoch
    average_loss = epoch_loss / batches

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.6f}"
    )

    # Save latest checkpoint
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss
        },
        LATEST_CHECKPOINT
    )

    # Save best checkpoint
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss
            },
            BEST_CHECKPOINT
        )
        print("New best model saved!")