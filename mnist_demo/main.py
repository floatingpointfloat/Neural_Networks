import pygame
import numpy as np
import torch

import render

from load_model import load_model
from predict import get_prediction_values

MATRIX_WIDTH = 500
MATRIX_HEIGHT = 500


def main():
    # Initialize Pygame and create the window.
    screen = render.initialize_render()

    # Load the trained model.
    model = load_model()

    # Select the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    print("Running on:", device)

    # Drawing matrix: 0 = black, 255 = white.
    matrix = np.zeros(
        (MATRIX_WIDTH, MATRIX_HEIGHT),
        dtype=np.uint8,
    )

    clock = pygame.time.Clock()

    probabilities = np.zeros(10, dtype=np.float32)
    guess = None

    running = True

    try:
        while running:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    # Reset the drawing and prediction.
                    if event.key == pygame.K_r:
                        matrix.fill(0)

                        render.OLD_POS_X = None
                        render.OLD_POS_Y = None

                        probabilities = np.zeros(
                            10,
                            dtype=np.float32,
                        )

                        guess = None

                    # Save the preprocessed 28x28 image.
                    elif event.key == pygame.K_p:
                        render.save_processed_image(matrix)

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    if event.button == 1:
                        render.OLD_POS_X = None
                        render.OLD_POS_Y = None

                        mouse_x, mouse_y = event.pos

                        inside_drawing_area = (
                            render.DRAW_X <= mouse_x < render.DRAW_X + MATRIX_WIDTH
                            and render.DRAW_Y <= mouse_y < render.DRAW_Y + MATRIX_HEIGHT
                        )

                        if inside_drawing_area:
                            matrix = render.update_matrix(
                                matrix,
                                MATRIX_WIDTH,
                                MATRIX_HEIGHT,
                            )

                elif event.type == pygame.MOUSEBUTTONUP:

                    if event.button == 1:

                        render.OLD_POS_X = None
                        render.OLD_POS_Y = None

                        # Predict only if the drawing is not empty.
                        if np.any(matrix):

                            predicted_digit, predicted_probabilities = (
                                get_prediction_values(
                                    model,
                                    device,
                                    matrix,
                                )
                            )

                            guess = int(predicted_digit.item())

                            probabilities = (
                                predicted_probabilities[0].detach().cpu().numpy()
                            )

            # Update the matrix while the left mouse button is held.
            if pygame.mouse.get_pressed()[0]:

                mouse_x, mouse_y = pygame.mouse.get_pos()

                inside_drawing_area = (
                    render.DRAW_X <= mouse_x < render.DRAW_X + MATRIX_WIDTH
                    and render.DRAW_Y <= mouse_y < render.DRAW_Y + MATRIX_HEIGHT
                )

                if inside_drawing_area:
                    matrix = render.update_matrix(
                        matrix,
                        MATRIX_WIDTH,
                        MATRIX_HEIGHT,
                    )

                else:
                    render.OLD_POS_X = None
                    render.OLD_POS_Y = None

            # Draw the interface.
            render.render_matrix(
                matrix,
                MATRIX_WIDTH,
                MATRIX_HEIGHT,
            )

            render.render_prediction_chart(
                probabilities,
                guess,
            )

            pygame.display.flip()

            clock.tick(120)

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
