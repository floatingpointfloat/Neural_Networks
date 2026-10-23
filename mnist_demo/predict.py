import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image


def matrix_to_tensor(matrix: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(matrix * 255)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    resized_matrix = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(resized_matrix)


def get_prediction_values(model, device, matrix: np.ndarray):
    tensor = matrix_to_tensor(matrix=matrix)

    probabilities = model(tensor).to(device)
    guess = torch.argmax(probabilities, dim=1)

    return guess, probabilities
