import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

from templates.model_template import SimpleValueNet as Model #has to change
from templates.config import DATASET_PATH

# Path to the checkpoint folder
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42

ALLOW_TRAINING = True

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.00001

VALIDATION_SPLIT = 0.1


def main():
    torch.backends.cudnn.benchmark = True #cuda optimization

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    #use CUDA if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Model().to(device)

    #master switch
    if not ALLOW_TRAINING:
        raise RuntimeError("No Training currently allowed. Please do not train the model")

    # Load the saved dataset
    data = torch.load(
        DATASET_PATH,
        map_location="cpu"
    )

    #can be ANY other name, you got change a few things
    train_data = data["train_data"]
    labels = data["labels"]

    dataset = TensorDataset(
        train_data,
        labels
    )

    # Split the dataset into training and validation data
    validation_size = int(len(dataset) * VALIDATION_SPLIT)
    training_size = len(dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"Training positions:   {len(train_dataset)}")
    print(f"Validation positions: {len(validation_dataset)}")

    # DataLoader for the training data
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, #cuda optimization
        pin_memory=True #faster gpu vram usage
    )

    # DataLoader for the validation data
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0, #cuda optimization
        pin_memory=True #faster gpu vram usage
    )

    #loss function - choose the right one haha
    criterion = torch.nn.MSELoss()

    #Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    #best validation loss - infinite at first before starting anything
    best_val_loss = float("inf")

    # Load the latest checkpoint if one exists
    if LATEST_CHECKPOINT.exists():
        print(f"Loading checkpoint: {LATEST_CHECKPOINT}")

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

        print(f"Resuming from epoch {start_epoch}")

        # Use the saved validation loss as the current best validation loss
        if "val_loss" in checkpoint:
            best_val_loss = checkpoint["val_loss"]

    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0

    #training loop:
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        #training
        model.train()

        epoch_loss = 0.0
        batches = 0

        for train_data, labels in train_loader:

            #move the data to the device (gpu cuda/cpu)
            train_data = train_data.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            #reset the gradients to 0
            optimizer.zero_grad()

            #forward pass
            predictions = model(train_data)

            # Remove the unnecessary dimension [128, 1] -> [128]
            predictions = predictions.squeeze(1)

            #loss calculation
            loss = criterion(
                predictions,
                labels
            )

            if batches % 1000 == 0:
                print(
                    f"Batch {batches} | "
                    f"Loss: {loss.item():.6f}"
                )

            #backpropagation aka pytorch does its magic
            loss.backward()

            #update the parameters
            optimizer.step()

            #track the loss
            epoch_loss += loss.item()
            batches += 1

        #average training loss for this epoch
        average_loss = epoch_loss / batches

        #validation

        model.eval()

        validation_loss = 0.0
        validation_batches = 0

        # Disable gradient calculation because we are not training
        with torch.no_grad():

            for validation_data, labels in validation_loader:

                #move the data to the device (gpu cuda/cpu)
                validation_data = validation_data.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                #forward pass
                predictions = model(validation_data)

                # Remove the unnecessary dimension [128, 1] -> [128]
                predictions = predictions.squeeze(1)

                #loss calculation
                loss = criterion(
                    predictions,
                    labels
                )

                #track the validation loss
                validation_loss += loss.item()
                validation_batches += 1

        #average validation loss for this epoch
        average_val_loss = validation_loss / validation_batches

        #print results
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Train Loss: {average_loss:.6f} "
            f"- Val Loss: {average_val_loss:.6f}"
        )


        #save latest checkpoint

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
                "val_loss": average_val_loss
            },
            LATEST_CHECKPOINT
        )


        #save best checkpoint

        if average_val_loss < best_val_loss:

            best_val_loss = average_val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": average_loss,
                    "val_loss": average_val_loss
                },
                BEST_CHECKPOINT
            )

            print("New best model saved!")


if __name__ == "__main__":
    main()