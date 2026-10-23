import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image


def matrix_to_tensor(matrix: np.ndarray) -> torch.Tensor:
    # Convert the drawing into an 8-bit image.
    matrix = matrix.astype(np.uint8)
    image = Image.fromarray(matrix * 255)

    # Find all pixels belonging to the digit.
    bbox = image.getbbox()

    # Handle an empty drawing.
    if bbox is None:
        empty_matrix = np.zeros((28, 28), dtype=np.float32)

        return torch.from_numpy(empty_matrix).unsqueeze(0).unsqueeze(0)

    # Crop the image to the digit's bounding box.
    image = image.crop(bbox)

    # Scale the digit while preserving its aspect ratio.
    width, height = image.size
    scale = 20 / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    # Create a black 28x28 canvas.
    canvas = Image.new("L", (28, 28), 0)

    # Center the digit on the canvas.
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas.paste(image, (x_offset, y_offset))

    # Convert pixel values from [0, 255] to [0, 1].
    resized_matrix = np.asarray(canvas, dtype=np.float32) / 255.0

    # Add batch and channel dimensions.
    tensor = torch.from_numpy(resized_matrix).unsqueeze(0).unsqueeze(0)

    return tensor


def get_prediction_values(model, device, matrix: np.ndarray):
    tensor = matrix_to_tensor(matrix)
    tensor = tensor.to(device)

    model.eval()

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        guess = torch.argmax(probabilities, dim=1)

    return guess, probabilities
