import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from paths import checkpoint_file
from model import CNN_ResNet_Model

EPOCHS = 100
BATCH_SIZE = 128
SAVE_EVERY_NTH_EPOCH = 1
LEARNING_RATE = 0.001
NUM_CLASSES = 10

RESUME_TRAINING = True

# master switch
ALLOW_TRAINING = True

if not ALLOW_TRAINING:
    raise RuntimeError("Training isn't allowed currently!")

# select the device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# define the model and stuff
model = CNN_ResNet_Model(num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_loss = float("inf")

if RESUME_TRAINING:  # load existing training data
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )

    # restore model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    if "epoch_complete" in checkpoint:
        if checkpoint["epoch_complete"]:
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = checkpoint["epoch"] - 1
    else:
        start_epoch = checkpoint["epoch"]

    if "best_loss" in checkpoint:
        best_loss = checkpoint["best_loss"]

    print("Resuming from epoch:", start_epoch + 1)

# dataset and dataloader
transform = transforms.ToTensor()  # convert images to tensors

DATA_DIR = r"C:\Users\Gigabyte\Documents\GitHub" r"\Neural_Networks\mnist_demo\data"

# train data
train_data = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

# test data
test_data = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# training loop
start_time = time.time()  # time tracking - preferably in hours

for epoch in range(start_epoch, start_epoch + EPOCHS):
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for b, (x_train, y_train) in enumerate(train_loader):
        b += 1  # just so our batches start at one - indices

        x_train = x_train.to(device)
        y_train = y_train.to(device)

        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * x_train.size(0)
        total_train_samples += x_train.size(0)

        if b % 50 == 0:
            print(
                f"-----Epoch {epoch + 1} \t|\t "
                f"Batch: {b} \t|\t Loss: {loss.item():.4f}-----"
            )

    avg_train_loss = total_train_loss / total_train_samples

    # testing
    model.eval()

    total_test_loss = 0.0
    total_test_samples = 0

    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
            b += 1  # same as above

            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_pred = model(x_test)
            loss = criterion(y_pred, y_test)

            total_test_loss += loss.item() * x_test.size(0)
            total_test_samples += x_test.size(0)

    avg_test_loss = total_test_loss / total_test_samples

    print(
        f"EPOCH [{epoch + 1}/{EPOCHS}] \t|\t "
        f"TRAIN LOSS: {avg_train_loss:.4f} \t|\t "
        f"TEST LOSS: {avg_test_loss:.4f}"
    )

    # save checkpoint
    if (epoch + 1) % SAVE_EVERY_NTH_EPOCH == 0 and avg_train_loss < best_loss:
        best_loss = avg_train_loss

        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "epoch_complete": True,
                "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_file,
        )
        print(f"Saved checkpoint at {checkpoint_file}\n")

# display the time it took
print(f"It took {(time.time() - start_time) / 60:.2f} minutes.")
