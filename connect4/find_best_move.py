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


@dataclass
class TTEntry:  # transposition table to check for exact same positions
    depth: int
    value: float
    flag: str


class SearchTimeout(Exception):
    pass

def evaluate(board, model, device):
    

# minimax search function - alpha beta search tree pruning
def minimax(
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
