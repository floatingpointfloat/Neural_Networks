import numpy as np
import torch

from PIL import Image

IMAGE_SIZE = 28
DIGIT_SIZE = 20


def matrix_to_image(matrix):
    """
    Convert the 500x500 drawing matrix into a 28x28 MNIST-style image.

    The drawing matrix uses:
        0   = black background
        255 = white drawing
    """

    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional drawing matrix.")

    # Convert to uint8 and transpose because the drawing matrix
    # uses [x, y], while PIL images use [y, x].
    matrix = np.clip(matrix, 0, 255).astype(np.uint8).T

    image = Image.fromarray(matrix)

    # Find the bounding box of the drawn digit.
    bbox = image.getbbox()

    # Return an empty MNIST image if nothing was drawn.
    if bbox is None:
        return Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    # Crop the empty surroundings.
    image = image.crop(bbox)

    width, height = image.size

    # Scale proportionally so the digit fits within 20x20 pixels.
    scale = DIGIT_SIZE / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    # Place the digit on a 28x28 black canvas.
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    x = (IMAGE_SIZE - new_width) // 2
    y = (IMAGE_SIZE - new_height) // 2

    canvas.paste(image, (x, y))

    # Center the digit approximately by its intensity-weighted
    # center of mass, similar to common MNIST preprocessing.
    pixels = np.asarray(canvas, dtype=np.float32)

    total_intensity = pixels.sum()

    if total_intensity > 0:
        y_coords, x_coords = np.indices(pixels.shape)

        center_x = (pixels * x_coords).sum() / total_intensity

        center_y = (pixels * y_coords).sum() / total_intensity

        target_x = (IMAGE_SIZE - 1) / 2
        target_y = (IMAGE_SIZE - 1) / 2

        shift_x = round(target_x - center_x)
        shift_y = round(target_y - center_y)

        # Move the image without wrapping pixels around the edges.
        shifted = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

        shifted.paste(canvas, (shift_x, shift_y))

        canvas = shifted

    return canvas


def matrix_to_tensor(matrix, device):
    """
    Convert the drawing matrix into a tensor of shape (1, 1, 28, 28).
    """

    image = matrix_to_image(matrix)

    # Convert pixel values from [0, 255] to [0, 1].
    array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array)

    # Add batch and channel dimensions.
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(device)


def get_prediction_values(model, device, matrix):
    """
    Return:
        predicted_character: predicted character as a string
    """

    CLASS_NAMES = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "n",
        "q",
        "r",
        "t",
    ]

    tensor = matrix_to_tensor(matrix, device)

    model.eval()

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(logits, dim=1)

        predicted_digit = torch.argmax(
            probabilities,
            dim=1,
        )

    predicted_index = predicted_digit.item()
    predicted_character = CLASS_NAMES[predicted_index]

    return predicted_character
