"""
Rules and general game state checks
-> for example draw, win, board
"""

import numpy as np
import torch

"""
Board structure: 
channel 0: pieces of the currently playing player 
channel1: pieces of the current opponent
channel 2: which player is currently playing, player 1 or 2?
"""
BOARD = np.zeros((3, 6, 7), dtype=np.int8)
PLAYER1 = 0
PLAYER2 = 1


def legal_moves(board):
    free_slots = []
    for column in range(7):
        if board[0, 0, column] == 0 and board[1, 0, column] == 0:
            free_slots.append(column)

    return free_slots


def make_move(board, column):
    if column not in legal_moves(board):
        raise RuntimeError(f"Invalid Move! Can't place a piece on column {column + 1}")

    height = np.sum((board[0, :, column] == 1) | (board[1, :, column] == 1))
    board[0, 5 - height, column] = 1

    return board


def flip_board(board):
    flipped = board.copy()

    flipped[0] = board[1]
    flipped[1] = board[0]
    flipped[2, :, :] = 1 - board[2, :, :]  # opposites player turn now :)

    return flipped
