import pygame
from sys import exit
import numpy as np
import torch

import render
from render import render_matrix, render_prediction_chart, update_matrix
from load_model import load_model
from predict import get_prediction_values

MATRIX_WIDTH, MATRIX_HEIGHT = 500, 500
matrix = np.zeros((MATRIX_WIDTH, MATRIX_HEIGHT), dtype=np.int8)

pygame.init()

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clock = pygame.time.Clock()

probabilities = np.zeros(10, dtype=np.float32)
guess = None

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                matrix = np.zeros(
                    (MATRIX_WIDTH, MATRIX_HEIGHT),
                    dtype=np.int8,
                )
                render.OLD_POS_X, render.OLD_POS_Y = None, None
                probabilities = np.zeros(10, dtype=np.float32)
                guess = None

        if event.type == pygame.MOUSEBUTTONUP:
            render.OLD_POS_X, render.OLD_POS_Y = None, None

            if event.button == 1 and np.any(matrix):
                predicted_digit, predicted_probabilities = get_prediction_values(
                    model,
                    device,
                    matrix,
                )

                guess = predicted_digit.item()
                probabilities = predicted_probabilities[0].cpu().numpy()

    if pygame.mouse.get_pressed()[0]:
        matrix = update_matrix(
            matrix,
            matrix_width=MATRIX_WIDTH,
            matrix_height=MATRIX_HEIGHT,
        )

    render_matrix(
        matrix,
        matrix_width=MATRIX_WIDTH,
        matrix_height=MATRIX_HEIGHT,
    )

    render_prediction_chart(probabilities, guess)

    pygame.display.flip()
    clock.tick(120)
