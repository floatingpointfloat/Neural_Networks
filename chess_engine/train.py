import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

from model import ChessValueNet
from config import DATASET_PATH

torch.backends.cudnn.benchmark = True #cuda optimization

ALLOW_TRAINING = True

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.00001

VALIDATION_SPLIT = 0.1

# Path to the checkpoint folder
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

def main():
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

    validation_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, #cuda optimization
        pin_memory=True #faster gpu vram usage
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2, #cuda optimization
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

    #best validation loss
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
        # Use the saved validation loss as the current best loss
        if "validation_loss" in checkpoint:
            best_loss = checkpoint["validation_loss"]
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0

    #training loop
    for epoch in range(start_epoch, EPOCHS + start_epoch):
        model.train()

        epoch_loss = 0.0
        batches = 0

        for boards, targets in train_loader:
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

        model.eval()

        validation_loss = 0.0
        validation_batches = 0

        with torch.no_grad():
            for boards, targets in validation_loader:
                boards = boards.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                predictions = model(boards)
                predictions = predictions.squeeze(1)

                loss = criterion(predictions, targets)

                validation_loss += loss.item()
                validation_batches += 1

        average_validation_loss = validation_loss / validation_batches

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.6f} "
            f"- Validation Loss: {average_validation_loss:.6f}"
        )

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
                "validation_loss": average_validation_loss
            },
            LATEST_CHECKPOINT
        )

        # Save best checkpoint
        if average_validation_loss < best_loss:
            best_loss = average_validation_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": average_loss,
                    "validation_loss": average_validation_loss
                },
                BEST_CHECKPOINT
            )
            print("New best model saved!")

if __name__ == "__main__":
    main()