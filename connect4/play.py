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
    with open("ai_beaten_amount.txt", "r") as f:
        ai_beaten_amount = f.read()
except FileNotFoundError:
    raise RuntimeError(
        "File ai_beaten_amount.txt wasn't found in this directory - might need to change the script from another directory."
    )

try:
    with open("human_beaten_amount.txt", "r") as f:
        human_beaten_amount = f.read()
except FileNotFoundError:
    raise RuntimeError(
        "File human_beaten_amount.txt wasn't found in this directory - might need to change the script from another directory."
    )

try:
    with open("games_played_amount.txt", "r") as f:
        human_beaten_amount = f.read()
except FileNotFoundError:
    raise RuntimeError(
        "File games_played_amount.txt wasn't found in this directory - might need to change the script from another directory."
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
    global board
    global current_player
    global game_over
    global last_move

    if game_over:
        return
    if ai_thinking:
        return
    if current_player != HUMAN:
        return
    if column not in legal_moves(board):
        window.set_status("That column is full!")
        return

    board = make_move(board, column)

    row = find_last_piece(column)
    last_move = (row, column)

    update_gui()

    finished, winner, draw = is_game_over(board)

    if finished:
        handle_game_over(winner, draw)
        return

    board = flip_board(board)
    current_player = AI

    update_gui()
    start_ai()


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
    global board
    global current_player
    global ai_thinking
    global game_over
    global last_move

    if not ai_thinking:
        return
    if not ai_result["finished"]:
        return

    ai_thinking = False

    move = ai_result["move"]
    value = ai_result["value"]

    print(f"AI move: {move}")
    print(f"AI value: {value:.4f}")

    if move is None:
        game_over = True
        window.set_status("AI found no move.")
        return

    board = make_move(board, move)

    row = find_last_piece(move)
    last_move = (row, move)

    update_gui()

    finished, winner, draw = is_game_over(board)

    if finished:
        handle_game_over(winner, draw)
        return

    board = flip_board(board)
    current_player = HUMAN
    update_gui()
    window.set_status("Your turn")


def handle_game_over(winner, draw):
    global game_over
    game_over = True

    if draw:
        window.set_status("Draw!")
    elif winner == HUMAN:
        window.set_status("You win!")
    elif winner == AI:
        window.set_status("AI wins!")


def new_game():
    global board
    global current_player
    global game_over
    global ai_thinking
    global last_move

    board = np.zeros((3, 6, 7), dtype=np.int8)

    current_player = HUMAN
    game_over = False
    ai_thinking = False
    last_move = None

    ai_result["finished"] = False
    ai_result["move"] = None
    ai_result["value"] = None

    window.set_status("Your turn")
    update_gui()


window.board_widget.move_requested.connect(human_move)
window.new_game_button.clicked.connect(new_game)


timer = QTimer()
timer.timeout.connect(check_ai_result)
timer.start(30)


update_gui()
sys.exit(app.exec())
