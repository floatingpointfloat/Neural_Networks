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

    original_alpha = alpha

    # transposition table

    key = board.tobytes()

    if transposition_table is not None and key in transposition_table:
        search_stats["tt_hits"] += 1
        entry = transposition_table[key]

        if entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.value
            elif entry.flag == "LOWERBOUND":
                alpha = max(alpha, entry.value)
            elif entry.flag == "UPPERBOUND":
                beta = min(beta, entry.value)

            if alpha >= beta:
                return entry.value

    game_over, winner, draw = is_game_over(board)

    if game_over:
        if draw:
            value = DRAW_VALUE
        current_player = get_current_player(board)
        if winner == current_player:
            value = WIN_VALUE
        value = LOSS_VALUE

        if transposition_table is not None:
            transposition_table[key] == TTEntry(depth=depth, value=value, flag="EXACT")

        return value

    if depth == 0:
        if is_critical_position(board):
            value = IMMEDIATE_LOSS

        value = evaluate(board, model, device)

        if transposition_table is not None:
            transposition_table[key] = TTEntry(depth=depth, value=value, flag="EXACT")

        return value

    """
    Actual Negamax Search
    """

    best_value = float("-inf")

    for move in order_moves(board):
        child = simulate_move(board, move)
        value = negamax(
            child,
            depth - 1,
            -beta,
            -alpha,
            model,
            device,
            time_limit,
            start_time,
            transposition_table,
            search_stats,
        )

        best_value = max(best_value, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break

        if best_value <= original_alpha:
            flag = "UPPERBOUND"
        elif best_value >= beta:
            flag = "LOWERBOUND"
        else:
            flag = "EXACT"

        # save to tt table
        if transposition_table is not None:
            transposition_table[key] = TTEntry(depth=depth, value=best_value, flag=flag)

        return best_value


def find_best_move(
    board,
    model,
    device,
    time_limit=TIMELIMIT,
    max_depth=20,
):
    start_time = time.time()

    best_move = None
    best_value = None

    current_depth = 1

    transposition_table = {}
    search_stats = {"tt_hits": 0}

    while current_depth <= max_depth:

        depth_best_move = None
        depth_best_value = float("-inf")
        depth_completed = True

        alpha = float("-inf")
        beta = float("+inf")

        for move in order_moves(board):

            child = simulate_move(board, move)

            try:
                value = -negamax(
                    child,
                    current_depth - 1,
                    -beta,
                    -alpha,
                    model,
                    device,
                    time_limit,
                    start_time,
                    transposition_table,
                    search_stats,
                )

            except SearchTimeout:
                depth_completed = False
                break

            if value > depth_best_value:
                depth_best_value = value
                depth_best_move = move

            alpha = max(alpha, value)

        # only use completed depths
        if depth_completed:
            best_move = depth_best_move
            best_value = depth_best_value

        # reached time limit
        if time.time() - start_time >= time_limit:
            break

        current_depth += 1

    print(f"Reached depth: {current_depth - 1}")
    print(f"TT Entries: {len(transposition_table)}")
    print(f"TT Hits: {search_stats['tt_hits']}")

    return best_move, best_value
