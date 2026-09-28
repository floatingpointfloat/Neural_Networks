#this is a test script and was made by chatgpt
import pygame
import torch
import chess
from pathlib import Path

from model import ChessValueNet
from find_best_move import find_best_move


# ============================================================
# Einstellungen
# ============================================================

DEPTH = 4

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


# ============================================================
# Gerät
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Modell laden
# ============================================================

model = ChessValueNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded.")


# ============================================================
# Pygame
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (BOARD_SIZE, BOARD_SIZE)
)

pygame.display.set_caption(
    "Chess Engine"
)

font = pygame.font.SysFont(
    "segoeuisymbol",
    64
)


# ============================================================
# Schachbrett
# ============================================================

board = chess.Board()

selected_square = None


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
# Brett zeichnen
# ============================================================

def draw_board():

    for row in range(8):

        for col in range(8):

            # Farbe des Feldes
            if (row + col) % 2 == 0:
                color = WHITE
            else:
                color = BROWN

            x = col * SQUARE_SIZE
            y = row * SQUARE_SIZE

            pygame.draw.rect(
                screen,
                color,
                (
                    x,
                    y,
                    SQUARE_SIZE,
                    SQUARE_SIZE
                )
            )

            # ------------------------------------------------
            # Ausgewähltes Feld
            # ------------------------------------------------

            square = chess.square(
                col,
                7 - row
            )

            if square == selected_square:

                pygame.draw.rect(
                    screen,
                    HIGHLIGHT,
                    (
                        x,
                        y,
                        SQUARE_SIZE,
                        SQUARE_SIZE
                    ),
                    5
                )

            # ------------------------------------------------
            # Figur zeichnen
            # ------------------------------------------------

            piece = board.piece_at(square)

            if piece is not None:

                symbol = pieces[
                    piece.symbol()
                ]

                text = font.render(
                    symbol,
                    True,
                    (0, 0, 0)
                )

                rect = text.get_rect(
                    center=(
                        x + SQUARE_SIZE // 2,
                        y + SQUARE_SIZE // 2
                    )
                )

                screen.blit(
                    text,
                    rect
                )


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

    return chess.square(
        col,
        7 - row
    )


# ============================================================
# Hauptschleife
# ============================================================

running = True

while running:

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        # ----------------------------------------------------
        # Mausklick
        # ----------------------------------------------------

        if event.type == pygame.MOUSEBUTTONDOWN:

            # KI ist Schwarz
            # Mensch ist Weiß

            if board.turn != chess.WHITE:
                continue

            if board.is_game_over():
                continue

            square = get_square_from_mouse(
                event.pos
            )

            if square is None:
                continue

            # ------------------------------------------------
            # Erstes Feld auswählen
            # ------------------------------------------------

            if selected_square is None:

                piece = board.piece_at(
                    square
                )

                if (
                    piece is not None
                    and piece.color == chess.WHITE
                ):

                    selected_square = square

            # ------------------------------------------------
            # Zweites Feld anklicken
            # ------------------------------------------------

            else:

                move = chess.Move(
                    selected_square,
                    square
                )

                # ------------------------------------------------
                # Bauern automatisch zur Dame
                # ------------------------------------------------

                piece = board.piece_at(
                    selected_square
                )

                if (
                    piece is not None
                    and piece.piece_type == chess.PAWN
                    and chess.square_rank(square) == 7
                ):

                    move = chess.Move(
                        selected_square,
                        square,
                        promotion=chess.QUEEN
                    )

                # ------------------------------------------------
                # Zug ausführen
                # ------------------------------------------------

                if move in board.legal_moves:

                    print(
                        "You:",
                        board.san(move)
                    )

                    board.push(move)

                    selected_square = None

                else:

                    # Ungültiger Zug
                    selected_square = None


    # ========================================================
    # KI
    # ========================================================

    if (
        board.turn == chess.BLACK
        and not board.is_game_over()
    ):

        print("AI thinking...")

        move, evaluation = find_best_move(
            board,
            DEPTH,
            model,
            device
        )

        if move is not None:

            print(
                "AI:",
                board.san(move),
                f"({evaluation:+.4f})"
            )

            board.push(move)


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