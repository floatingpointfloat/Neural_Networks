import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image


def matrix_to_image(matrix: np.ndarray) -> Image.Image:
    matrix = np.asarray(matrix, dtype=np.uint8)

    image = Image.fromarray(matrix * 255)
    bbox = image.getbbox()

    canvas = Image.new("L", (28, 28), 0)

    if bbox is None:
        return canvas

    image = image.crop(bbox)

    width, height = image.size
    scale = 20.0 / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas.paste(image, (x_offset, y_offset))

    pixels = np.asarray(canvas, dtype=np.float32)
    total_intensity = pixels.sum()

    if total_intensity > 0:
        y_coords, x_coords = np.indices(pixels.shape)

        center_x = (pixels * x_coords).sum() / total_intensity
        center_y = (pixels * y_coords).sum() / total_intensity

        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))

        shifted = Image.new("L", (28, 28), 0)
        shifted.paste(canvas, (shift_x, shift_y))
        canvas = shifted

    return canvas


def matrix_to_tensor(matrix: np.ndarray) -> torch.Tensor:
    image = matrix_to_image(matrix)

    resized_matrix = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    tensor = torch.from_numpy(resized_matrix).unsqueeze(0).unsqueeze(0)

    return tensor


def get_prediction_values(model, device, matrix: np.ndarray):
    tensor = matrix_to_tensor(matrix).to(device)

    model.eval()

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        guess = torch.argmax(probabilities, dim=1)

    return guess, probabilities
