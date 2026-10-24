import torch
import time

from paths import checkpoint_file
from model import CNN_ResNet_Model

NUM_CLASSES = 47


def load_model():
    # measuring time
    start_time = time.time()

    # select the device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # define the model and stuff
    model = CNN_ResNet_Model(num_classes=NUM_CLASSES).to(device)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )

    # restore model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # evaluation mode - IMPORTANT!

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(
        f"Succesfully loaded the model.\n-----Loading took {round(time.time() - start_time, 1)}seconds"
    )
    return model


if __name__ == "__main__":
    model = load_model()
