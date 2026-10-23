from sys import exit
import numpy as np
import pygame
import torch

from load_model import load_model
from predict import get_prediction_values

WIDTH, HEIGHT = 600, 700
BORDER_WIDTH = 10
OLD_POS_X, OLD_POS_Y = None, None

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a number! | Press 'r' to reset")
clock = pygame.time.Clock()

# Select device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()

# Fonts
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 20)

# Prediction values
probabilities = np.zeros(10, dtype=np.float32)
guess = None


def render_matrix(matrix: np.ndarray, matrix_width=500, matrix_height=500) -> None:
    pixels_per_element = min(
        (WIDTH - 100) / matrix_width,
        (HEIGHT - 200) / matrix_height,
    )

    render_width = matrix_width * pixels_per_element
    render_height = matrix_height * pixels_per_element

    matrix_x = (WIDTH - render_width) // 2
    matrix_y = 30

    screen.fill((0, 0, 0))

    pygame.draw.rect(
        screen,
        (255, 0, 0),
        (
            matrix_x - BORDER_WIDTH,
            matrix_y - BORDER_WIDTH,
            render_width + 2 * BORDER_WIDTH,
            render_height + 2 * BORDER_WIDTH,
        ),
        width=BORDER_WIDTH,
        border_radius=10,
    )  # render the boundary

    for x in range(matrix_width):
        for y in range(matrix_height):
            if matrix[x, y] == 1:
                pygame.draw.rect(
                    screen,
                    (255, 255, 255),
                    (
                        matrix_x + x * pixels_per_element,
                        matrix_y + y * pixels_per_element,
                        pixels_per_element,
                        pixels_per_element,
                    ),
                )


def render_prediction_chart(probabilities: np.ndarray, guess) -> None:
    chart_x = 55
    chart_y = 570
    chart_width = 490
    chart_height = 80

    title = font.render("Prediction probabilities", True, (255, 255, 255))
    screen.blit(title, (chart_x, 560))

    # Chart axes
    pygame.draw.line(
        screen,
        (180, 180, 180),
        (chart_x, chart_y + chart_height),
        (chart_x + chart_width, chart_y + chart_height),
        2,
    )

    bar_gap = 10
    bar_width = (chart_width - 9 * bar_gap) / 10

    for digit, probability in enumerate(probabilities):
        bar_height = int(float(probability) * chart_height)
        bar_x = int(chart_x + digit * (bar_width + bar_gap))
        bar_y = chart_y + chart_height - bar_height

        # Highlight the predicted digit
        if guess is not None and digit == guess:
            bar_color = (255, 60, 60)
        else:
            bar_color = (190, 190, 190)

        pygame.draw.rect(
            screen,
            bar_color,
            (
                bar_x,
                bar_y,
                int(bar_width),
                bar_height,
            ),
        )

        # Digit labels
        label = small_font.render(str(digit), True, (255, 255, 255))
        screen.blit(
            label,
            (
                bar_x + (bar_width - label.get_width()) / 2,
                chart_y + chart_height + 5,
            ),
        )

        # Probability labels
        percentage = small_font.render(
            f"{probability * 100:.0f}%",
            True,
            (255, 255, 255),
        )

        label_y = max(chart_y - 20, bar_y - 20)

        screen.blit(
            percentage,
            (bar_x + (bar_width - percentage.get_width()) / 2, label_y),
        )

    if guess is not None:
        prediction_text = font.render(
            f"Prediction: {guess}",
            True,
            (255, 60, 60),
        )
        screen.blit(prediction_text, (400, 560))


def draw_brush(matrix: np.ndarray, x: int, y: int, radius: int) -> None:
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx**2 + dy**2 <= radius**2:
                px = x + dx
                py = y + dy

                if 0 <= px < matrix.shape[0] and 0 <= py < matrix.shape[1]:
                    matrix[px, py] = 1


def update_matrix(matrix: np.ndarray, matrix_width=500, matrix_height=500) -> None:
    global OLD_POS_X, OLD_POS_Y

    pixels_per_element = min(
        (WIDTH - 100) / matrix_width,
        (HEIGHT - 200) / matrix_height,
    )

    render_width = matrix_width * pixels_per_element
    render_height = matrix_height * pixels_per_element

    matrix_x = (WIDTH - render_width) // 2
    matrix_y = 30

    pos_x, pos_y = pygame.mouse.get_pos()

    if (
        pos_x < matrix_x
        or pos_x >= matrix_x + render_width
        or pos_y < matrix_y
        or pos_y >= matrix_y + render_height
    ):
        OLD_POS_X, OLD_POS_Y = None, None
        return matrix

    x = int((pos_x - matrix_x) / pixels_per_element)
    y = int((pos_y - matrix_y) / pixels_per_element)

    if OLD_POS_X is not None and OLD_POS_Y is not None:
        distance = max(abs(x - OLD_POS_X), abs(y - OLD_POS_Y))

        for i in range(distance + 1):
            t = i / max(distance, 1)

            new_x = round(OLD_POS_X + (x - OLD_POS_X) * t)
            new_y = round(OLD_POS_Y + (y - OLD_POS_Y) * t)

            draw_brush(matrix, new_x, new_y, radius=20)
    else:
        draw_brush(matrix, x, y, radius=20)

    OLD_POS_X, OLD_POS_Y = x, y

    return matrix


if __name__ == "__main__":
    matrix_width = 500
    matrix_height = 500
    test_matrix = np.zeros((matrix_width, matrix_height), dtype=np.int8)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    test_matrix = np.zeros(
                        (matrix_width, matrix_height),
                        dtype=np.int8,
                    )
                    OLD_POS_X, OLD_POS_Y = None, None
                    probabilities = np.zeros(10, dtype=np.float32)
                    guess = None

            if event.type == pygame.MOUSEBUTTONUP:
                OLD_POS_X, OLD_POS_Y = None, None

                # Predict after finishing the drawing
                if event.button == 1 and np.any(test_matrix):
                    predicted_digit, predicted_probabilities = get_prediction_values(
                        model,
                        device,
                        test_matrix,
                    )

                    guess = predicted_digit.item()
                    probabilities = predicted_probabilities[0].cpu().numpy()

        # get constant actions
        mouse_buttons = pygame.mouse.get_pressed()

        if mouse_buttons[0]:
            test_matrix = update_matrix(
                test_matrix,
                matrix_width=matrix_width,
                matrix_height=matrix_height,
            )

        render_matrix(
            test_matrix,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
        )

        render_prediction_chart(probabilities, guess)

        pygame.display.flip()
        clock.tick(120)
