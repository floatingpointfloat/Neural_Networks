import pygame
import torch
import numpy as np
import threading

from find_best_move import find_best_move
from gamestates import (
    legal_moves,
    make_move,
    flip_board,
    is_game_over,
)

# ============================================================
# Einstellungen
# ============================================================

WIDTH = 700
HEIGHT = 650

ROWS = 6
COLS = 7
CELL_SIZE = 90

FPS = 60

# Maximale Denkzeit der KI
TIME_LIMIT = 5

# Maximale Suchtiefe
MAX_DEPTH = 20


# Spieler
HUMAN = 0
AI = 1


# ============================================================
# Farben
# ============================================================

BACKGROUND = (30, 30, 30)
BOARD_COLOR = (30, 80, 180)

EMPTY_COLOR = (240, 240, 240)

# Mensch = IMMER Rot
PLAYER_COLOR = (220, 50, 50)

# KI = IMMER Gelb
AI_COLOR = (240, 210, 40)

TEXT_COLOR = (255, 255, 255)


# ============================================================
# Board erstellen
# ============================================================


def create_board():

    board = np.zeros((3, 6, 7), dtype=np.int8)

    # Mensch beginnt
    board[2, :, :] = HUMAN

    return board


# ============================================================
# Zug ausführen
# ============================================================


def apply_move(board, column):

    new_board = board.copy()

    make_move(new_board, column)

    # Perspektive für die KI wechseln
    return flip_board(new_board)


# ============================================================
# Board zeichnen
# ============================================================


def draw_board(screen, board, thinking):

    screen.fill(BACKGROUND)

    # --------------------------------------------------------
    # Spielfeld
    # --------------------------------------------------------

    pygame.draw.rect(screen, BOARD_COLOR, (0, 60, COLS * CELL_SIZE, ROWS * CELL_SIZE))

    # --------------------------------------------------------
    # Aktuellen Spieler bestimmen
    # --------------------------------------------------------

    current_player = board[2, 0, 0]

    # --------------------------------------------------------
    # Steine zeichnen
    # --------------------------------------------------------

    for row in range(ROWS):

        for col in range(COLS):

            x = col * CELL_SIZE + CELL_SIZE // 2

            y = 60 + row * CELL_SIZE + CELL_SIZE // 2

            color = EMPTY_COLOR

            # ------------------------------------------------
            # Channel 0 = aktueller Spieler
            # ------------------------------------------------

            if board[0, row, col] == 1:

                if current_player == HUMAN:
                    color = PLAYER_COLOR

                else:
                    color = AI_COLOR

            # ------------------------------------------------
            # Channel 1 = Gegner
            # ------------------------------------------------

            elif board[1, row, col] == 1:

                if current_player == HUMAN:
                    color = AI_COLOR

                else:
                    color = PLAYER_COLOR

            # ------------------------------------------------
            # Stein zeichnen
            # ------------------------------------------------

            pygame.draw.circle(screen, color, (x, y), CELL_SIZE // 2 - 8)

    # --------------------------------------------------------
    # Spaltennummern
    # --------------------------------------------------------

    font = pygame.font.SysFont(None, 32)

    for col in range(COLS):

        text = font.render(str(col + 1), True, TEXT_COLOR)

        screen.blit(text, (col * CELL_SIZE + 38, 15))

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status_font = pygame.font.SysFont(None, 28)

    if thinking:

        text = "AI is thinking..."

    else:

        text = "Your turn"

    rendered = status_font.render(text, True, TEXT_COLOR)

    screen.blit(rendered, (500, 15))


# ============================================================
# AI Worker
# ============================================================


def ai_worker(board, model, device, result):

    # Die eigentliche Suche läuft hier komplett
    # außerhalb des Pygame-Threads.

    move, value = find_best_move(
        board,
        model,
        device,
        time_limit=TIME_LIMIT,
        max_depth=MAX_DEPTH,
    )

    result["move"] = move
    result["value"] = value
    result["finished"] = True


# ============================================================
# Main
# ============================================================


def main():

    pygame.init()

    # --------------------------------------------------------
    # Fenster
    # --------------------------------------------------------

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Connect Four - Neural Network")

    clock = pygame.time.Clock()

    # --------------------------------------------------------
    # Model laden
    # --------------------------------------------------------

    print("Loading model...")

    from load_model import load_model

    model = load_model()

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Spielzustand
    # --------------------------------------------------------

    board = create_board()

    running = True

    game_over = False
    winner = None
    draw = False

    human_turn = True

    thinking = False

    ai_thread = None

    # Ergebnis des AI-Threads
    ai_result = {
        "finished": False,
        "move": None,
        "value": None,
    }

    # ========================================================
    # Game Loop
    # ========================================================

    while running:

        # ====================================================
        # Events
        # ====================================================

        for event in pygame.event.get():

            # ------------------------------------------------
            # Fenster schließen
            # ------------------------------------------------

            if event.type == pygame.QUIT:

                running = False

            # ------------------------------------------------
            # Mensch macht einen Zug
            # ------------------------------------------------

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and human_turn
                and not thinking
                and not game_over
            ):

                mouse_x = event.pos[0]

                column = mouse_x // CELL_SIZE

                # Prüfen, ob der Zug erlaubt ist
                if column in legal_moves(board):

                    # Zug ausführen
                    board = apply_move(board, column)

                    # Spielstatus prüfen
                    game_over, winner, draw = is_game_over(board)

                    # Falls das Spiel noch läuft,
                    # ist jetzt die KI dran
                    if not game_over:

                        human_turn = False

        # ====================================================
        # AI-Suche starten
        # ====================================================

        if not human_turn and not thinking and not game_over:

            print("Starting AI search...")

            # ------------------------------------------------
            # WICHTIG:
            # Eigene Board-Kopie für den Suchthread
            # ------------------------------------------------

            search_board = board.copy()

            # Neues Ergebnisobjekt
            ai_result = {
                "finished": False,
                "move": None,
                "value": None,
            }

            # ------------------------------------------------
            # AI Thread
            # ------------------------------------------------

            ai_thread = threading.Thread(
                target=ai_worker,
                args=(
                    search_board,
                    model,
                    device,
                    ai_result,
                ),
                daemon=True,
            )

            thinking = True

            ai_thread.start()

        # ====================================================
        # Prüfen, ob AI fertig ist
        # ====================================================

        if thinking and ai_result["finished"]:

            move = ai_result["move"]
            value = ai_result["value"]

            # ------------------------------------------------
            # Sicherheit
            # ------------------------------------------------

            if move is not None:

                print(f"AI move: {move + 1} " f"| value: {value:.3f}")

                # ------------------------------------------------
                # AI-Zug auf das echte Board anwenden
                # ------------------------------------------------

                board = apply_move(board, move)

                # ------------------------------------------------
                # Spielstatus prüfen
                # ------------------------------------------------

                game_over, winner, draw = is_game_over(board)

            # ------------------------------------------------
            # AI fertig
            # ------------------------------------------------

            thinking = False

            human_turn = True

            ai_thread = None

        # ====================================================
        # Board rendern
        # ====================================================

        draw_board(screen, board, thinking)

        # ====================================================
        # Game Over anzeigen
        # ====================================================

        if game_over:

            font = pygame.font.SysFont(None, 55)

            if draw:

                text = "DRAW!"

            elif winner == HUMAN:

                text = "YOU WIN!"

            else:

                text = "AI WINS!"

            rendered = font.render(text, True, TEXT_COLOR)

            screen.blit(rendered, (WIDTH // 2 - rendered.get_width() // 2, 580))

        # ====================================================
        # Bildschirm aktualisieren
        # ====================================================

        pygame.display.flip()

        # ====================================================
        # 60 FPS
        # ====================================================

        clock.tick(FPS)

    # ========================================================
    # Beenden
    # ========================================================

    pygame.quit()


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    main()
