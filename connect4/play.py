import sys
import threading
import numpy as np
from PySide6.QtCore import QTimer

from gui import create_app
from gamestates import (
    legal_moves,
    make_move,
    flip_board,
    is_game_over,
)
from find_best_move import find_best_move
from load_model import load_model

TIME_LIMIT = 5

HUMAN = 0
AI = 1


try:
    with open("scores.txt", "r") as f:
        """
        Structure:
        line 1 - human_beaten_amount
        line 2 - ai_beaten_amount
        line 3 - amount_draws
        line 4 - total_games_played
        """
        lines = f.readlines()

        human_beaten_amount = int(lines[0].strip())
        ai_beaten_amount = int(lines[1].strip())
        amount_draws = int(lines[2].strip())
        total_games_played = int(lines[3].strip())

except FileNotFoundError:
    raise RuntimeError(
        "File scores.txt wasn't found in this directory - might need to change the script from another directory."
    )
except (IndexError, ValueError):
    raise RuntimeError(
        "File scores.txt was found, but its structure isn't as expected."
    )

app, window = create_app()

model = load_model()
model.eval()

device = next(model.parameters()).device

board = np.zeros((3, 6, 7), dtype=np.int8)

current_player = HUMAN

game_over = False
ai_thinking = False
last_move = None
pending_move = None

ai_result = {
    "finished": False,
    "move": None,
    "value": None,
}


def update_gui():
    window.board_widget.set_board(board, last_move)
    window.board_widget.set_human_turn(current_player == HUMAN)
    window.board_widget.set_ai_turn(current_player == AI)


def find_last_piece(column):
    for row in range(6):
        if board[0, row, column] == 1 or board[1, row, column] == 1:
            return row

    return None


def human_move(column):
    global pending_move

    if game_over:
        return
    if ai_thinking:
        return
    if current_player != HUMAN:
        return
    if column not in legal_moves(board):
        window.set_status("That column is full!")
        return
    pending_move = column
    window.board_widget.animate_move(column, HUMAN)


def ai_worker(search_board):
    move, value = find_best_move(search_board, model, device, time_limit=TIME_LIMIT)
    ai_result["move"] = move
    ai_result["value"] = value
    ai_result["finished"] = True


def start_ai():
    global ai_thinking

    ai_thinking = True
    window.set_status("AI is thinking...")
    update_gui()
    search_board = board.copy()

    ai_result["finished"] = False
    ai_result["move"] = None
    ai_result["value"] = None

    thread = threading.Thread(target=ai_worker, args=(search_board,), daemon=True)
    thread.start()


def check_ai_result():
    global ai_thinking
    global pending_move

    if not ai_thinking:
        return
    if not ai_result["finished"]:
        return

    move = ai_result["move"]
    value = ai_result["value"]

    print(f"AI move: {move}")
    print(f"AI value: {value:.4f}")

    if move is None:
        ai_thinking = False
        finish_ai_no_move()
        return
    pending_move = move
    ai_thinking = False

    # Prevent the timer from starting the animation again
    ai_result["finished"] = False
    window.board_widget.animate_move(move, AI)


def finish_animation():
    if pending_move is None:
        return
    if current_player == HUMAN:
        finish_human_move()
    elif current_player == AI:
        finish_ai_move()


def finish_human_move():
    global board
    global current_player
    global game_over
    global last_move
    global pending_move

    column = pending_move
    board = make_move(board, column)
    row = find_last_piece(column)
    last_move = (row, column)
    pending_move = None

    update_gui()

    finished, winner, draw = is_game_over(board)
    if finished:
        handle_game_over(winner, draw)
        return
    board = flip_board(board)
    current_player = AI

    update_gui()
    start_ai()


def finish_ai_move():
    global board
    global current_player
    global game_over
    global last_move
    global pending_move
    move = pending_move

    board = make_move(board, move)
    row = find_last_piece(move)
    last_move = (row, move)
    pending_move = None
    update_gui()
    finished, winner, draw = is_game_over(board)

    if finished:
        handle_game_over(winner, draw)
        return
    board = flip_board(board)
    current_player = HUMAN

    update_gui()
    window.set_status("Your turn")


def finish_ai_no_move():
    global game_over
    game_over = True
    window.set_status("AI found no move.")


def handle_game_over(winner, draw):
    global game_over
    global human_beaten_amount
    global ai_beaten_amount
    global amount_draws
    global total_games_played

    game_over = True

    if draw:
        window.set_status("Draw!")
        amount_draws += 1
    elif winner == HUMAN:
        window.set_status("You win!")
        ai_beaten_amount += 1
    elif winner == AI:
        window.set_status("AI wins!")
        human_beaten_amount += 1
    total_games_played += 1

    with open("scores.txt", "w", encoding="utf-8") as f:
        f.write(f"{human_beaten_amount}\n")
        f.write(f"{ai_beaten_amount}\n")
        f.write(f"{amount_draws}\n")
        f.write(f"{total_games_played}\n")

    window.set_score_label(
        f"Games played total: {total_games_played} | "
        f"Times lost: {human_beaten_amount} | "
        f"Times won: {ai_beaten_amount} | "
        f"Draws: {amount_draws}"
    )


def new_game():
    global board
    global current_player
    global game_over
    global ai_thinking
    global last_move
    global pending_move

    board = np.zeros((3, 6, 7), dtype=np.int8)

    current_player = HUMAN

    game_over = False
    ai_thinking = False
    last_move = None
    pending_move = None
    ai_result["finished"] = False
    ai_result["move"] = None
    ai_result["value"] = None
    window.set_status("Your turn")

    update_gui()


window.board_widget.move_requested.connect(human_move)
window.board_widget.animation_finished.connect(finish_animation)
window.new_game_button.clicked.connect(new_game)
window.set_score_label(
    f"Games played total: {total_games_played} | "
    f"Times lost: {human_beaten_amount} | "
    f"Times won: {ai_beaten_amount} | "
    f"Draws: {amount_draws}"
)


timer = QTimer()
timer.timeout.connect(check_ai_result)
timer.start(30)


new_game()
sys.exit(app.exec())
