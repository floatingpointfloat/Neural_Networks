import sys
import threading

import numpy as np

from PySide6.QtCore import QTimer

from gui import run_gui

from gamestates import (
    legal_moves,
    make_move,
    flip_board,
    check_win,
    check_draw,
    is_game_over,
    get_current_player,
    simulate_move,
)

from find_best_move import find_best_move
from load_model import load_model

# --------------------------------------------------
# Einstellungen
# --------------------------------------------------

TIME_LIMIT = 5
MAX_DEPTH = 20

HUMAN = 0
AI = 1


# --------------------------------------------------
# Modell
# --------------------------------------------------

model = load_model()
model.eval()


# --------------------------------------------------
# Spielzustand
# --------------------------------------------------

board = np.zeros((3, 6, 7), dtype=np.int8)

current_player = HUMAN

game_over = False
ai_thinking = False

ai_result = {
    "finished": False,
    "move": None,
    "value": None,
}


# --------------------------------------------------
# AI Thread
# --------------------------------------------------


def ai_worker(search_board):

    global ai_result

    move, value = find_best_move(
        search_board,
        model,
        next(iter(model.parameters())).device,
        time_limit=TIME_LIMIT,
        max_depth=MAX_DEPTH,
    )

    ai_result["move"] = move
    ai_result["value"] = value
    ai_result["finished"] = True


# --------------------------------------------------
# Menschlicher Zug
# --------------------------------------------------


def human_move(column):

    global board
    global current_player
    global ai_thinking
    global game_over

    if game_over:
        return

    if current_player != HUMAN:
        return

    if ai_thinking:
        return

    if column not in legal_moves(board):
        return

    # Zug ausführen
    board = make_move(board, column)

    window.set_board(board)

    # Prüfen
    finished, winner, draw = is_game_over(board)

    if finished:
        game_over = True

        if draw:
            print("Unentschieden!")
        else:
            print(f"Spieler {winner} gewinnt!")

        return

    # Perspektive wechseln
    board = flip_board(board)

    current_player = AI

    window.set_board(board)

    start_ai()


# --------------------------------------------------
# AI starten
# --------------------------------------------------


def start_ai():

    global ai_thinking

    if game_over:
        return

    ai_thinking = True

    ai_result["finished"] = False
    ai_result["move"] = None
    ai_result["value"] = None

    window.set_ai_thinking(True)
    window.set_human_turn(False)

    search_board = board.copy()

    thread = threading.Thread(
        target=ai_worker,
        args=(search_board,),
        daemon=True,
    )

    thread.start()


# --------------------------------------------------
# AI Ergebnis überprüfen
# --------------------------------------------------


def check_ai_result():

    global board
    global current_player
    global ai_thinking
    global game_over

    if not ai_thinking:
        return

    if not ai_result["finished"]:
        return

    move = ai_result["move"]
    value = ai_result["value"]

    ai_thinking = False

    window.set_ai_thinking(False)

    print(f"AI spielt Spalte {move + 1}" f" | Value: {value:.4f}")

    if move is None:
        print("AI konnte keinen Zug finden.")
        game_over = True
        return

    if move not in legal_moves(board):
        print("FEHLER: AI hat illegalen Zug gewählt!")
        game_over = True
        return

    # AI-Zug
    board = make_move(board, move)

    window.set_board(board)

    # Prüfen
    finished, winner, draw = is_game_over(board)

    if finished:

        game_over = True

        if draw:
            print("Unentschieden!")
        else:
            print(f"Spieler {winner} gewinnt!")

        return

    # Perspektive zurück auf Mensch
    board = flip_board(board)

    current_player = HUMAN

    window.set_board(board)
    window.set_human_turn(True)


# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------

app, window = run_gui(board)

window.move_requested.connect(human_move)


# Timer läuft im GUI-Thread.
# Dadurch bleibt das Fenster während
# der AI-Berechnung responsiv.

timer = QTimer()
timer.timeout.connect(check_ai_result)
timer.start(30)


print("Connect Four gestartet.")
print("Du bist Rot.")
print("Klicke auf eine Spalte.")


sys.exit(app.exec())
