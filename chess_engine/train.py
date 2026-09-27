import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import ChessDataset
from model import ChessValueNet
from config import PGN_PATH

ALLOW_TRAINING = True
MAX_GAMES = 10
MIN_ELO = 2000

BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 0.0001

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


dataset = ChessDataset(
    pgn_path=PGN_PATH,
    max_games=MAX_GAMES,
    min_elo=MIN_ELO
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE
)

# Check the target distribution - only for debug
white_wins = 0
draws = 0
black_wins = 0

for boards, targets in loader:

    white_wins += (targets == 1.0).sum().item()
    draws += (targets == 0.0).sum().item()
    black_wins += (targets == -1.0).sum().item()


total = white_wins + draws + black_wins

print(f"White wins: {white_wins} ({white_wins / total * 100:.2f}%)")
print(f"Draws:      {draws} ({draws / total * 100:.2f}%)")
print(f"Black wins: {black_wins} ({black_wins / total * 100:.2f}%)")


#loss function
criterion = torch.nn.MSELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#best loss
best_loss = float("inf")

# Load the latest checkpoint if one exists
if LATEST_CHECKPOINT.exists():

    print(
        f"Loading checkpoint: {LATEST_CHECKPOINT}"
    )

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    print(
        f"Resuming from epoch {start_epoch}"
    )


    # Use the saved loss as the current best loss
    if "loss" in checkpoint:
        best_loss = checkpoint["loss"]

else:

    print("No checkpoint found. Starting from scratch.")

#training loop
for epoch in range(EPOCHS):
    model.train()

    epoch_loss = 0.0
    batches = 0

    for boards, targets in loader:
        #move the data to the devide (gpu cuda/cpu)
        boards = boards.to(device)
        targets = targets.to(device)

        #reset the gradients to 0
        optimizer.zero_grad()

        #forward pass
        predictions = model(boards)

        #only for debug
        with torch.no_grad():
            if batches % 500 == 0:
                print(
                    f"Batch {batches} | "
                    f"Prediction: mean={predictions.mean().item():.6f}, "
                    f"std={predictions.std().item():.6f}, "
                    f"min={predictions.min().item():.6f}, "
                    f"max={predictions.max().item():.6f}"
                )

                print(
                    f"Batch {batches} | "
                    f"Target:     mean={targets.mean().item():.6f}, "
                    f"std={targets.std().item():.6f}, "
                    f"min={targets.min().item():.6f}, "
                    f"max={targets.max().item():.6f}"
                )

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