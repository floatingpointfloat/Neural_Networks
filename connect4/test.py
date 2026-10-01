import numpy as np
import torch
import time

from find_best_move import find_best_move
from gamestates import (
    legal_moves,
    make_move,
    flip_board,
    is_game_over,
    get_current_player,
)

# ============================================================
# Configuration
# ============================================================

TIME_LIMIT = 5
MAX_DEPTH = 15

HUMAN = 0
AI = 1


# ============================================================
# Board
# ============================================================


def create_board():
    """
    Creates an empty Connect Four board.

    Channel 0 = current player
    Channel 1 = opponent
    Channel 2 = actual player whose turn it is
    """

    board = np.zeros((3, 6, 7), dtype=np.int8)

    # Player 0 starts
    board[2, :, :] = 0

    return board


# ============================================================
# Display
# ============================================================


def print_board(board):
    print()

    for row in range(6):

        line = ""

        for column in range(7):

            if board[0, row, column] == 1:
                line += "X "

            elif board[1, row, column] == 1:
                line += "O "

            else:
                line += ". "

        print(line)

    print("1 2 3 4 5 6 7")
    print()


# ============================================================
# Human move
# ============================================================


def get_human_move(board):
    """
    Ask the human for a legal column.
    """

    legal = legal_moves(board)

    while True:

        try:
            move = int(input("Your move (1-7): ")) - 1

        except ValueError:
            print("Please enter a number from 1 to 7.")
            continue

        if move not in legal:
            print("That column is full or invalid.")
            continue

        return move


# ============================================================
# Apply move
# ============================================================


def apply_move(board, move):
    """
    Applies a move and changes perspective.

    The internal representation always stores:

        channel 0 = current player
        channel 1 = opponent

    Therefore after a move we flip the board.
    """

    new_board = board.copy()

    make_move(new_board, move)

    return flip_board(new_board)


# ============================================================
# Main game
# ============================================================


def play_game():

    print("=" * 70)
    print("CONNECT FOUR - HUMAN VS AI")
    print("=" * 70)

    print()
    print("You are X.")
    print("AI is O.")
    print("Enter columns using numbers 1-7.")
    print()

    model = None

    # Load model
    from load_model import load_model

    model = load_model()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print()
    print(f"Device: {device}")
    print()

    board = create_board()

    move_number = 1

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    ai_moves = 0
    total_ai_time = 0.0
    total_ai_depth = 0

    # --------------------------------------------------------
    # Game loop
    # --------------------------------------------------------

    while True:

        print_board(board)

        # ----------------------------------------------------
        # Check game over
        # ----------------------------------------------------

        game_over, winner, draw = is_game_over(board)

        if game_over:

            print_board(board)

            if draw:
                print("DRAW!")

            elif winner == HUMAN:
                print("YOU WIN!")

            else:
                print("AI WINS!")

            break

        # ----------------------------------------------------
        # Human turn
        # ----------------------------------------------------

        current_player = get_current_player(board)

        if current_player == HUMAN:

            print(f"Move {move_number}")
            print("Your turn.")

            move = get_human_move(board)

            print(f"You play column {move + 1}")

            board = apply_move(board, move)

        # ----------------------------------------------------
        # AI turn
        # ----------------------------------------------------

        else:

            print(f"Move {move_number}")
            print("AI is thinking...")

            start = time.perf_counter()

            move, value = find_best_move(
                board,
                model,
                device,
                time_limit=TIME_LIMIT,
                max_depth=MAX_DEPTH,
            )

            elapsed = time.perf_counter() - start

            ai_moves += 1
            total_ai_time += elapsed

            print()
            print("----------------------------------------")
            print("AI RESULT")
            print("----------------------------------------")
            print(f"Move       : {move + 1}")
            print(f"Evaluation : {value:.4f}")
            print(f"Time       : {elapsed:.3f}s")
            print("----------------------------------------")

            board = apply_move(board, move)

        move_number += 1

    # ========================================================
    # Statistics
    # ========================================================

    print()
    print("=" * 70)
    print("GAME STATISTICS")
    print("=" * 70)

    print(f"Total moves : {move_number - 1}")
    print(f"AI moves    : {ai_moves}")

    if ai_moves > 0:

        print(f"Average AI time : " f"{total_ai_time / ai_moves:.3f}s")

    print("=" * 70)


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    play_game()
