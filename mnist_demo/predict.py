import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image


def matrix_to_tensor(matrix: np.ndarray) -> torch.Tensor:
    matrix = matrix.astype(np.uint8)  # prevent errors
    image = Image.fromarray(matrix * 255)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    resized_matrix = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(resized_matrix).unsqueeze(0).unsqueeze(0)
    # double unsqueeze cause the model thinks it will get something like this:
    # [Batch, Channels, Height, Width]
    # [1,     1,        28,     28]


def get_prediction_values(model, device, matrix: np.ndarray):
    tensor = matrix_to_tensor(matrix=matrix)
    tensor = tensor.to(
        device
    )  # push the tensor to the device - preferably the gpu lol (will be in the debug in the terminal when starting the model)

    model.eval()
    with torch.inference_mode():  # don't record these measurements - no gradients have to be calculated
        logits = model(tensor)
        probabilities = F.softmax(
            logits, dim=1
        )  # convert the logits into probabilities
        guess = torch.argmax(probabilities, dim=1)

    return guess, probabilities
