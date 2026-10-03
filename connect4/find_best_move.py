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

IMMEDIATE_LOSS = -0.999

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
    original_beta = beta

    key = board.tobytes()

    if transposition_table is not None and key in transposition_table:
        entry = transposition_table[key]

        if search_stats is not None:
            search_stats["tt_hits"] += 1

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

        else:
            current_player = get_current_player(board)
            if winner == current_player:
                value = WIN_VALUE
            else:
                value = LOSS_VALUE

        if transposition_table is not None:
            transposition_table[key] = TTEntry(
                depth=depth,
                value=value,
                flag="EXACT",
            )

        return value

    if depth == 0:
        if is_critical_position(board):
            value = IMMEDIATE_LOSS
        else:
            value = evaluate(board, model, device)
        if transposition_table is not None:
            transposition_table[key] = TTEntry(
                depth=depth,
                value=value,
                flag="EXACT",
            )
        return value

    best_value = float("-inf")

    for move in order_moves(board):
        child = simulate_move(board, move)
        value = -negamax(
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
        # alpha-beta cutoff
        if alpha >= beta:
            break

    # tt flags
    if best_value <= original_alpha:
        flag = "UPPERBOUND"
    elif best_value >= original_beta:
        flag = "LOWERBOUND"
    else:
        flag = "EXACT"

    # save tt
    if transposition_table is not None:
        transposition_table[key] = TTEntry(
            depth=depth,
            value=best_value,
            flag=flag,
        )
    return best_value


def find_best_move(
    board,
    model,
    device,
    time_limit=TIMELIMIT,
    max_depth=20,
):
    start_time = time.time()  # manual test also is part of the time
    moves = order_moves(board)
    if not moves:
        return None, DRAW_VALUE

    # Check for an immediate winning move
    for move in moves:
        child = simulate_move(board, move)
        game_over, winner, draw = is_game_over(child)
        if game_over and not draw:
            current_player = get_current_player(child)
            if winner != current_player:
                print(f"Immediate winning move found: {move}")
                return move, WIN_VALUE
    # check for immediate opponent winning moves
    opponent_board = flip_board(board)
    for opponent_move in order_moves(opponent_board):
        child = simulate_move(opponent_board, opponent_move)
        game_over, winner, draw = is_game_over(child)
        if game_over and not draw:
            current_player = get_current_player(child)
            if winner != current_player:
                print(
                    f"Immediate Opponent winning move found ({opponent_move}) - blocking..."
                )
                return opponent_move, LOSS_VALUE
    # check for situations where the opponent could win via a ai move (like a staircase situation typa thingy)
    safe_moves = moves.copy()
    for move in moves:
        ai_child = simulate_move(board, move)
        for opponent_move in order_moves(ai_child):
            opponent_child = simulate_move(ai_child, opponent_move)
            game_over, winner, draw = is_game_over(opponent_child)
            if game_over and not draw:
                current_player = get_current_player(opponent_child)
                if winner != current_player:
                    safe_moves.remove(move)
                    break
    # in case there are no safe moves - just to be more explicit
    if not safe_moves:
        safe_moves = moves

    # Use the first legal move as a fallback
    best_move = moves[0]
    best_value = DRAW_VALUE

    current_depth = 1

    transposition_table = {}
    search_stats = {"tt_hits": 0}

    while current_depth <= max_depth:

        depth_best_move = None
        depth_best_value = float("-inf")
        depth_completed = True

        alpha = float("-inf")
        beta = float("+inf")

        for move in safe_moves:
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

        # Only use completed depths
        if depth_completed:
            best_move = depth_best_move
            best_value = depth_best_value

        # Reached time limit
        if time.time() - start_time >= time_limit:
            break

        current_depth += 1

    print(f"Reached depth: {current_depth - 1}")
    print(f"TT Entries: {len(transposition_table)}")
    print(f"TT Hits: {search_stats['tt_hits']}")

    return best_move, best_value
