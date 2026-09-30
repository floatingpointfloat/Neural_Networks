"""
Rules and general game state checks
-> for example draw, win, board
"""

import numpy as np
import torch

BOARD = np.zeros((3, 6, 7), dtype=np.int8)
PLAYER1 = 1.0
PLAYER2 = 0.0


def legal_moves(board):
    free_slots = []
    for column in range(7):
        if board[0, 0, column] == 0 and board[1, 0, column] == 0:
            free_slots.append(column)

    return free_slots
