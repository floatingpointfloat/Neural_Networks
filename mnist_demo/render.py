import pygame
import numpy as np

# Window settings
WINDOW_WIDTH, WINDOW_HEIGHT = 650, 720

# Drawing settings
MATRIX_WIDTH, MATRIX_HEIGHT = 500, 500
DRAW_X, DRAW_Y = 20, 20
BRUSH_RADIUS = 15

# Colors
BACKGROUND = (25, 25, 30)
DRAW_BACKGROUND = (0, 0, 0)
DRAW_COLOR = (255, 255, 255)

TEXT_COLOR = (235, 235, 240)
BAR_BACKGROUND = (60, 60, 70)
BAR_COLOR = (70, 180, 110)

# Previous mouse position
OLD_POS_X, OLD_POS_Y = None, None


def initialize_render():
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    pygame.display.set_caption("MNIST Digit Recognition")

    return screen


def update_matrix(
    matrix,
    matrix_width=MATRIX_WIDTH,
    matrix_height=MATRIX_HEIGHT,
):
    global OLD_POS_X, OLD_POS_Y

    mouse_x, mouse_y = pygame.mouse.get_pos()

    # Convert window coordinates to matrix coordinates.
    x = mouse_x - DRAW_X
    y = mouse_y - DRAW_Y

    # Only draw inside the drawing area.
    if not (0 <= x < matrix_width and 0 <= y < matrix_height):
        OLD_POS_X, OLD_POS_Y = None, None
        return matrix

    # Remember the previous position.
    if OLD_POS_X is None or OLD_POS_Y is None:
        OLD_POS_X, OLD_POS_Y = x, y

    # Draw a continuous line between the old and new position.
    distance = max(
        abs(x - OLD_POS_X),
        abs(y - OLD_POS_Y),
    )

    steps = max(1, distance)

    for i in range(steps + 1):
        t = i / steps

        px = round(OLD_POS_X + (x - OLD_POS_X) * t)
        py = round(OLD_POS_Y + (y - OLD_POS_Y) * t)

        # Limit the brush area to the matrix boundaries.
        x_min = max(0, px - BRUSH_RADIUS)
        x_max = min(matrix_width, px + BRUSH_RADIUS + 1)

        y_min = max(0, py - BRUSH_RADIUS)
        y_max = min(matrix_height, py + BRUSH_RADIUS + 1)

        # Create a circular brush mask.
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

        mask = (xx - px) ** 2 + (yy - py) ** 2 <= BRUSH_RADIUS**2

        # Update the original matrix in place.
        matrix[x_min:x_max, y_min:y_max][mask] = 255

    OLD_POS_X, OLD_POS_Y = x, y

    return matrix


def render_matrix(
    matrix,
    matrix_width=MATRIX_WIDTH,
    matrix_height=MATRIX_HEIGHT,
):
    screen = pygame.display.get_surface()

    if screen is None:
        return

    screen.fill(BACKGROUND)

    # Draw the black drawing area.
    pygame.draw.rect(
        screen,
        DRAW_BACKGROUND,
        (
            DRAW_X,
            DRAW_Y,
            matrix_width,
            matrix_height,
        ),
    )

    # Convert the matrix into an RGB surface.
    rgb = np.repeat(
        matrix[:, :, np.newaxis],
        3,
        axis=2,
    )

    drawing_surface = pygame.surfarray.make_surface(rgb)

    screen.blit(
        drawing_surface,
        (DRAW_X, DRAW_Y),
    )

    # Draw a border around the drawing area.
    pygame.draw.rect(
        screen,
        (100, 100, 110),
        (
            DRAW_X,
            DRAW_Y,
            matrix_width,
            matrix_height,
        ),
        2,
    )


def render_prediction_chart(probabilities, guess):
    screen = pygame.display.get_surface()

    if screen is None:
        return

    font = pygame.font.SysFont("Arial", 23)
    small_font = pygame.font.SysFont("Arial", 17)

    # Display the predicted digit.
    if guess is None:
        prediction_text = "Prediction: -"
    else:
        prediction_text = f"Prediction: {guess}"

    text_surface = font.render(
        prediction_text,
        True,
        TEXT_COLOR,
    )

    screen.blit(
        text_surface,
        (20, 535),
    )

    # Display the probability chart.
    title_surface = small_font.render(
        "Class probabilities",
        True,
        TEXT_COLOR,
    )

    screen.blit(
        title_surface,
        (20, 570),
    )

    bar_x = 70
    bar_width = 440
    bar_height = 9

    for digit in range(10):
        y = 595 + digit * 12

        # Digit label.
        label = small_font.render(
            str(digit),
            True,
            TEXT_COLOR,
        )

        screen.blit(
            label,
            (40, y - 5),
        )

        # Background bar.
        pygame.draw.rect(
            screen,
            BAR_BACKGROUND,
            (
                bar_x,
                y,
                bar_width,
                bar_height,
            ),
        )

        # Probability bar.
        probability = float(probabilities[digit])

        pygame.draw.rect(
            screen,
            BAR_COLOR,
            (
                bar_x,
                y,
                int(bar_width * probability),
                bar_height,
            ),
        )

        # Percentage label.
        percentage = small_font.render(
            f"{probability * 100:.1f}%",
            True,
            TEXT_COLOR,
        )

        screen.blit(
            percentage,
            (bar_x + bar_width + 10, y - 5),
        )
