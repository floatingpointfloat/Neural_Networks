"""
Load the parameters of the from steven0226 trained model into this fitting one. Function to call in other scripts to make everything ccleaner :)
"""

import torch
from safetensors.torch import load_file

from model import ValueNet as Model


def load_model():
    print("\nLoading pretrained model...")

    model = Model(blocks=6, filters=96)

    # choose the device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the weights and everything into the model
    weights = load_file("model.safetensors")
    own_weights = (
        model.state_dict()
    )  # only load the suitable weights (this model misses the policy head, so we don't need the parameters of that)

    loaded = []
    missing = []

    for name, target in own_weights.items():
        if name in weights and weights[name].shape == target.shape:
            own_weights[name] = weights[name]
            loaded.append(name)
        else:
            missing.append(name)

    model.load_state_dict(own_weights)
    model.to(device=device)
    model.eval()

    print("\nModel loaded successfully.")
    print(f"Used parameter groups: {len(loaded)}")
    print(f"Unused layers: {missing}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Trainable parameters (already trained): {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
    )

    return model


if __name__ == "__main__":
    load_model()
