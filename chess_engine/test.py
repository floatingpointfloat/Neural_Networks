# this is a test script and was made by chatgpt

import pygame
import torch
import chess
from pathlib import Path
import threading

from model import ChessValueNet
from find_best_move import find_best_move

# ============================================================
# Einstellungen
# ============================================================

AI_TIME_LIMIT = 5  # in seconds

BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoints" / "best.pt"


# ============================================================
# Farben
# ============================================================

WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

HIGHLIGHT = (255, 255, 100)

# Dein letzter Zug
PLAYER_MOVE_COLOR = (80, 200, 80)

# KI-Zug
AI_MOVE_COLOR = (80, 150, 255)

# Text
TEXT_COLOR = (255, 255, 255)


# ============================================================
# Gerät
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Modell laden
# ============================================================

model = ChessValueNet().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded.")


# ============================================================
# Pygame
# ============================================================

pygame.init()

screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))

pygame.display.set_caption("Chess Engine")


# Figuren-Font
font = pygame.font.SysFont("segoeuisymbol", 64)

# Status-Font
status_font = pygame.font.SysFont("arial", 24, bold=True)


# ============================================================
# Schachbrett
# ============================================================

board = chess.Board()

selected_square = None


# ============================================================
# Zug-Markierungen
# ============================================================

# Felder des letzten Spielerzuges
player_move_from = None
player_move_to = None

# Felder des letzten KI-Zuges
ai_move_from = None
ai_move_to = None


# ============================================================
# KI-Status
# ============================================================

ai_thinking = False

ai_result = None

ai_thread = None


# ============================================================
# Schachfiguren
# ============================================================

pieces = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


# ============================================================
# KI-Funktion
# ============================================================


def calculate_ai_move():

    global ai_result
    global ai_thinking

    print()
    print("AI thinking...")

    move, evaluation = find_best_move(
        board.copy(), model, device, time_limit=AI_TIME_LIMIT
    )

    ai_result = (move, evaluation)

    ai_thinking = False


# ============================================================
# Brett zeichnen
# ============================================================


def draw_board():

    for row in range(8):

        for col in range(8):

            # ------------------------------------------------
            # Farbe des Feldes
            # ------------------------------------------------

            if (row + col) % 2 == 0:
                color = WHITE
            else:
                color = BROWN

            x = col * SQUARE_SIZE
            y = row * SQUARE_SIZE

            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

            # ------------------------------------------------
            # Schachfeld
            # ------------------------------------------------

            square = chess.square(col, 7 - row)

            # ------------------------------------------------
            # Spielerzug markieren
            # ------------------------------------------------

            if square == player_move_from or square == player_move_to:

                pygame.draw.rect(
                    screen,
                    PLAYER_MOVE_COLOR,
                    (x + 4, y + 4, SQUARE_SIZE - 8, SQUARE_SIZE - 8),
                    6,
                )

            # ------------------------------------------------
            # KI-Zug markieren
            # ------------------------------------------------

            if square == ai_move_from or square == ai_move_to:

                pygame.draw.rect(
                    screen,
                    AI_MOVE_COLOR,
                    (x + 4, y + 4, SQUARE_SIZE - 8, SQUARE_SIZE - 8),
                    6,
                )

            # ------------------------------------------------
            # Ausgewähltes Feld
            # ------------------------------------------------

            if square == selected_square:

                pygame.draw.rect(screen, HIGHLIGHT, (x, y, SQUARE_SIZE, SQUARE_SIZE), 5)

            # ------------------------------------------------
            # Figur zeichnen
            # ------------------------------------------------

            piece = board.piece_at(square)

            if piece is not None:

                symbol = pieces[piece.symbol()]

                text = font.render(symbol, True, (0, 0, 0))

                rect = text.get_rect(
                    center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)
                )

                screen.blit(text, rect)

    # ========================================================
    # Status anzeigen
    # ========================================================

    if ai_thinking:

        status = "AI denkt..."

        text = status_font.render(status, True, TEXT_COLOR)

        background = pygame.Surface((text.get_width() + 20, text.get_height() + 10))

        background.set_alpha(180)

        screen.blit(background, (10, 10))

        screen.blit(text, (20, 15))


# ============================================================
# Mausposition -> Schachfeld
# ============================================================


def get_square_from_mouse(position):

    x, y = position

    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE

    if not (0 <= col < 8):
        return None

    if not (0 <= row < 8):
        return None

    return chess.square(col, 7 - row)


# ============================================================
# Hauptschleife
# ============================================================

running = True


while running:

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    for event in pygame.event.get():

        # ----------------------------------------------------
        # Fenster schließen
        # ----------------------------------------------------

        if event.type == pygame.QUIT:

            running = False

        # ----------------------------------------------------
        # Mausklick
        # ----------------------------------------------------

        if event.type == pygame.MOUSEBUTTONDOWN:

            # ------------------------------------------------
            # Während die KI denkt, keine Züge erlauben
            # ------------------------------------------------

            if ai_thinking:
                continue

            # ------------------------------------------------
            # KI ist Schwarz
            # Mensch ist Weiß
            # ------------------------------------------------

            if board.turn != chess.WHITE:
                continue

            if board.is_game_over():
                continue

            square = get_square_from_mouse(event.pos)

            if square is None:
                continue

            # ------------------------------------------------
            # Erstes Feld auswählen
            # ------------------------------------------------

            if selected_square is None:

                piece = board.piece_at(square)

                if piece is not None and piece.color == chess.WHITE:

                    selected_square = square

            # ------------------------------------------------
            # Zweites Feld anklicken
            # ------------------------------------------------

            else:

                move = chess.Move(selected_square, square)

                # ------------------------------------------------
                # Bauern automatisch zur Dame
                # ------------------------------------------------

                piece = board.piece_at(selected_square)

                if (
                    piece is not None
                    and piece.piece_type == chess.PAWN
                    and chess.square_rank(square) == 7
                ):

                    move = chess.Move(selected_square, square, promotion=chess.QUEEN)

                # ------------------------------------------------
                # Zug ausführen
                # ------------------------------------------------

                if move in board.legal_moves:

                    print("You:", board.san(move))

                    # --------------------------------------------
                    # Alten KI-Zug löschen
                    # --------------------------------------------

                    ai_move_from = None
                    ai_move_to = None

                    # --------------------------------------------
                    # Spielerzug speichern
                    # --------------------------------------------

                    player_move_from = move.from_square
                    player_move_to = move.to_square

                    # --------------------------------------------
                    # Zug ausführen
                    # --------------------------------------------

                    board.push(move)

                    selected_square = None

                    # --------------------------------------------
                    # KI vorbereiten
                    # --------------------------------------------

                    ai_result = None

                    ai_thinking = True

                    # --------------------------------------------
                    # KI in eigenem Thread starten
                    # --------------------------------------------

                    ai_thread = threading.Thread(target=calculate_ai_move, daemon=True)

                    ai_thread.start()

                else:

                    # Ungültiger Zug
                    selected_square = None

    # ========================================================
    # Ergebnis der KI überprüfen
    # ========================================================

    if (
        not ai_thinking
        and ai_result is not None
        and board.turn == chess.BLACK
        and not board.is_game_over()
    ):

        move, evaluation = ai_result

        ai_result = None

        if move is not None:

            print("AI:", board.san(move), f"({evaluation:+.4f})")

            # ------------------------------------------------
            # KI-Zug markieren
            # ------------------------------------------------

            ai_move_from = move.from_square
            ai_move_to = move.to_square

            # ------------------------------------------------
            # Zug ausführen
            # ------------------------------------------------

            board.push(move)

        else:

            print("AI could not find a move.")

    # ========================================================
    # Zeichnen
    # ========================================================

    draw_board()

    pygame.display.flip()


# ============================================================
# Ende
# ============================================================

pygame.quit()

print()
print("Game over!")
print("Result:", board.result())
