import torch
import numpy as np
import time
import random
from dataclasses import dataclass


from load_model import load_model
from gamestates import (
    legal_moves,
    make_move,
    flip_board,
    check_draw,
    check_win,
    is_game_over,
    get_current_player,
    copy_board,
    simulate_move,
    order_moves,
    is_critical_position,
)

TIMELIMIT = 10  # in seconds haha

WIN_VALUE = 1.0
LOSS_VALUE = -1.0
DRAW_VALUE = 0.0

IMMEDIATE_LOSS = -0.99

"""
Board structure: 
channel 0: pieces of the currently playing player 
channel1: pieces of the current opponent
channel 2: which player is currently playing, player 1 or 2?
"""
BOARD = np.zeros((3, 6, 7), dtype=np.int8)
PLAYER1 = 0
PLAYER2 = 1

model = load_model()
model.eval()


@dataclass
class TTEntry:  # transposition table to check for exact same positions
    depth: int
    value: float
    flag: str


class SearchTimeout(Exception):
    pass


def evaluate(board, model, device):
    x = torch.from_numpy(board).float()
    x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        value = model(x)

    return value.item()


# minimax search function - alpha beta search tree pruning
def negamax(
    board,
    depth,
    alpha,
    beta,
    model,
    device,
    time_limit=TIMELIMIT,
    start_time=None,
    transposition_table=None,
    search_stats=None,
):

    if time.time() - start_time > time_limit:
        raise SearchTimeout

    game_over, winner, draw = is_game_over(board)

    if game_over:
        if draw:
            return DRAW_VALUE
        current_player = get_current_player(board)
        if winner == current_player:
            return WIN_VALUE
        return LOSS_VALUE

    if depth == 0:
        if is_critical_position(board):
            return IMMEDIATE_LOSS

        return evaluate(board, model, device)

    # transposition table

    key = board.tobytes()

    if transposition_table is not None and key in transposition_table:
        entry = transposition_table[key]

        if entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.value
            if entry.flag == "LOWERBOUND":
                alpha = max(alpha, entry.value)
