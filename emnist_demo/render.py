from sys import exit
import numpy as np
import pygame

WIDTH, HEIGHT = 600, 700
BORDER_WIDTH = 10
OLD_POS_X, OLD_POS_Y = None, None

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a number! | Press 'r' to reset")
clock = pygame.time.Clock()


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

    pygame.display.flip()


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

            draw_brush(matrix, new_x, new_y, radius=5)
    else:
        draw_brush(matrix, x, y, radius=5)

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
                    test_matrix = np.zeros((matrix_width, matrix_height), dtype=np.int8)
                    OLD_POS_X, OLD_POS_Y = None, None
            if event.type == pygame.MOUSEBUTTONUP:
                OLD_POS_X, OLD_POS_Y = None, None

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

        clock.tick(120)
