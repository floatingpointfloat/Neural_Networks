import torch
import numpy as np

from load_model import load_model

ROWS = 6
COLS = 7


# ============================================================
# BOARD REPRESENTATION
# ============================================================
#
# Internes Board:
#
# board[0] = Steine von Spieler 1
# board[1] = Steine von Spieler 2
# board[2] = Turn
#
# Turn:
#   0 = Spieler 1
#   1 = Spieler 2
#
#
# Netzwerk-Input:
#
# channel 0 = Steine des aktuellen Spielers
# channel 1 = Steine des Gegners
# channel 2 = Spieler am Zug
# ============================================================


def create_board():
    """
    Erstellt ein leeres Connect-4-Board.
    """

    return np.zeros(
        (3, ROWS, COLS),
        dtype=np.float32,
    )


def legal_moves(board):
    """
    Gibt alle Spalten zurück, in die noch ein Stein
    gespielt werden kann.
    """

    moves = []

    for col in range(COLS):

        # Eine Spalte ist frei, wenn das oberste Feld
        # noch von keinem Spieler belegt ist.
        if board[0, 0, col] == 0 and board[1, 0, col] == 0:
            moves.append(col)

    return moves


def make_move(board, col):
    """
    Spielt einen Zug in der angegebenen Spalte.

    Gibt eine Kopie des Boards zurück.
    Das ursprüngliche Board wird nicht verändert.
    """

    new_board = board.copy()

    turn = int(board[2, 0, 0])

    # Von unten nach oben nach dem ersten freien Feld suchen.
    for row in range(ROWS - 1, -1, -1):

        if new_board[0, row, col] == 0 and new_board[1, row, col] == 0:
            new_board[turn, row, col] = 1.0
            break

    # Spieler wechseln.
    new_board[2, :, :] = 1 - turn

    return new_board


# ============================================================
# NETWORK INPUT
# ============================================================


def board_for_network(board):
    """
    Wandelt unser internes Board in exakt die Darstellung um,
    mit der das Netzwerk trainiert wurde.

    Netzwerk:

        channel 0 = aktueller Spieler
        channel 1 = Gegner
        channel 2 = Spieler am Zug
    """

    turn = int(board[2, 0, 0])

    network_board = np.zeros_like(board)

    if turn == 0:

        # Spieler 1 ist am Zug.
        network_board[0] = board[0]
        network_board[1] = board[1]

    else:

        # Spieler 2 ist am Zug.
        network_board[0] = board[1]
        network_board[1] = board[0]

    network_board[2] = turn

    return torch.tensor(
        network_board,
        dtype=torch.float32,
    )


# ============================================================
# VALUE PREDICTION
# ============================================================


def predict(model, board, device):
    """
    Berechnet den Value eines Boards.

    Der zurückgegebene Value ist immer aus Sicht
    des Spielers, der aktuell am Zug ist.
    """

    network_board = board_for_network(board)

    network_board = network_board.unsqueeze(0).to(device)

    with torch.no_grad():
        value = model(network_board)

    return value.item()


# ============================================================
# BOARD PRINTING
# ============================================================


def print_board(board):
    """
    Gibt das interne Board schön formatiert aus.
    """

    turn = int(board[2, 0, 0])

    print("    0   1   2   3   4   5   6")
    print("  +---+---+---+---+---+---+---+")

    for row in range(ROWS):

        line = "  |"

        for col in range(COLS):

            if board[0, row, col] == 1:
                symbol = "X"

            elif board[1, row, col] == 1:
                symbol = "O"

            else:
                symbol = " "

            line += f" {symbol} |"

        print(line)
        print("  +---+---+---+---+---+---+---+")

    print()
    print(f"Player {turn + 1} to move")


# ============================================================
# NETWORK INPUT PRINTING
# ============================================================


def print_network_input(board):
    """
    Zeigt die drei Kanäle so an, wie sie das Netzwerk erhält.
    """

    network_board = board_for_network(board).numpy()

    current_player = int(board[2, 0, 0]) + 1
    opponent = 3 - current_player

    print()
    print("NETWORK INPUT")
    print("-" * 60)

    print(f"Channel 0 = Current player (Player {current_player})")

    for row in range(ROWS):

        line = ""

        for col in range(COLS):

            if network_board[0, row, col] == 1:
                line += "X "

            else:
                line += ". "

        print(line)

    print()

    print(f"Channel 1 = Opponent (Player {opponent})")

    for row in range(ROWS):

        line = ""

        for col in range(COLS):

            if network_board[1, row, col] == 1:
                line += "O "

            else:
                line += ". "

        print(line)

    print()

    print(f"Channel 2 = Turn ({current_player})")

    print()


# ============================================================
# VALUE FROM BOTH PERSPECTIVES
# ============================================================


def evaluate_both_perspectives(model, board, device):
    """
    Bewertet eine Stellung aus beiden Spielerperspektiven.

    Wichtig:

    Das Netzwerk selbst bekommt immer die Perspektive
    des Spielers am Zug.

    Deshalb erzeugen wir zwei Boards:

        1. Originalstellung
        2. Gleiche Stellung, aber mit anderem Spieler
           am Zug

    Damit können wir überprüfen, ob das Netzwerk
    die Perspektive korrekt verarbeitet.
    """

    # --------------------------------------------------------
    # Originale Perspektive
    # --------------------------------------------------------

    original_turn = int(board[2, 0, 0])

    value_original = predict(
        model,
        board,
        device,
    )

    # --------------------------------------------------------
    # Gleiche Stellung mit umgedrehtem Spieler am Zug
    # --------------------------------------------------------

    reversed_board = board.copy()

    reversed_turn = 1 - original_turn

    reversed_board[2, :, :] = reversed_turn

    value_reversed = predict(
        model,
        reversed_board,
        device,
    )

    return value_original, value_reversed


# ============================================================
# SINGLE MOVE ANALYSIS
# ============================================================


def analyze_move(
    model,
    board,
    col,
    device,
    move_number,
):
    """
    Analysiert einen einzelnen möglichen Zug.
    """

    current_player = int(board[2, 0, 0])

    next_player = 1 - current_player

    print()
    print("=" * 75)
    print(f"MOVE {move_number}")
    print("=" * 75)

    print()
    print(f"Player {current_player + 1} plays column {col}")

    # --------------------------------------------------------
    # Zug ausführen
    # --------------------------------------------------------

    new_board = make_move(
        board,
        col,
    )

    # --------------------------------------------------------
    # Resultierendes Board
    # --------------------------------------------------------

    print()
    print("RESULTING BOARD")
    print("-" * 75)

    print_board(new_board)

    # --------------------------------------------------------
    # Netzwerk-Input
    # --------------------------------------------------------

    print_network_input(new_board)

    # --------------------------------------------------------
    # Value aus Sicht des nächsten Spielers
    # --------------------------------------------------------

    next_player_value = predict(
        model,
        new_board,
        device,
    )

    print()
    print("-" * 75)

    print(
        f"Value from Player {next_player + 1} perspective:" f" {next_player_value:+.4f}"
    )

    # --------------------------------------------------------
    # Perspektive umdrehen
    # --------------------------------------------------------

    other_board = new_board.copy()

    other_board[2, :, :] = current_player

    current_player_value = predict(
        model,
        other_board,
        device,
    )

    print(
        f"Value from Player {current_player + 1} perspective:"
        f" {current_player_value:+.4f}"
    )

    # --------------------------------------------------------
    # Summe / Symmetrie prüfen
    # --------------------------------------------------------

    perspective_sum = next_player_value + current_player_value

    print()
    print(f"Perspective sum:" f" {perspective_sum:+.4f}")

    print("Expected:" " values should ideally have approximately" " opposite signs.")

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print("INTERPRETATION")
    print("-" * 75)

    if next_player_value > 0.5:

        print(
            f"Network strongly prefers the position" f" for Player {next_player + 1}."
        )

    elif next_player_value < -0.5:

        print(
            f"Network strongly dislikes the position" f" for Player {next_player + 1}."
        )

    else:

        print("Network sees the position as relatively balanced.")

    return new_board


# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    print()

    # --------------------------------------------------------
    # Model laden
    # --------------------------------------------------------

    model = load_model()

    print()

    # ========================================================
    # TEST POSITION
    # ========================================================

    board = create_board()

    # Wir spielen eine kleine echte Spielsequenz:
    #
    # P1 -> 3
    # P2 -> 2
    # P1 -> 3
    # P2 -> 4
    # P1 -> 3
    # P2 -> 4
    #
    # Danach ist P1 am Zug.
    # ========================================================

    moves = [
        3,
        2,
        3,
        4,
        3,
        4,
    ]

    print("=" * 75)
    print("CREATING TEST POSITION")
    print("=" * 75)

    print()

    for move_number, col in enumerate(
        moves,
        start=1,
    ):

        player = int(board[2, 0, 0])

        print(f"Move {move_number}:" f" Player {player + 1}" f" -> column {col}")

        board = make_move(
            board,
            col,
        )

    # ========================================================
    # AKTUELLE STELLUNG
    # ========================================================

    print()
    print("=" * 75)
    print("CURRENT POSITION")
    print("=" * 75)

    print()

    print_board(board)

    # ========================================================
    # CURRENT VALUE
    # ========================================================

    current_player = int(board[2, 0, 0])

    current_value = predict(
        model,
        board,
        device,
    )

    print()
    print(f"Current value for Player {current_player + 1}:" f" {current_value:+.4f}")

    # ========================================================
    # NETWORK INPUT
    # ========================================================

    print_network_input(board)

    # ========================================================
    # LEGAL MOVES
    # ========================================================

    moves = legal_moves(board)

    print()
    print("=" * 75)
    print("LEGAL MOVES")
    print("=" * 75)

    print()

    print(f"Player {current_player + 1} has" f" {len(moves)} legal moves:")

    print(" ".join(str(move) for move in moves))

    # ========================================================
    # ALLE ZÜGE ANALYSIEREN
    # ========================================================

    print()
    print("=" * 75)
    print("ANALYZING EVERY LEGAL MOVE")
    print("=" * 75)

    for move_number, col in enumerate(
        moves,
        start=1,
    ):

        analyze_move(
            model=model,
            board=board,
            col=col,
            device=device,
            move_number=move_number,
        )

    # ========================================================
    # ZUSAMMENFASSUNG
    # ========================================================

    print()
    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print()

    print(
        f"{'Column':<10}" f"{'Next player value':<25}" f"{'Original player value':<25}"
    )

    print("-" * 60)

    for col in moves:

        new_board = make_move(
            board,
            col,
        )

        next_player_value = predict(
            model,
            new_board,
            device,
        )

        original_perspective = new_board.copy()

        original_perspective[2, :, :] = current_player

        original_player_value = predict(
            model,
            original_perspective,
            device,
        )

        print(
            f"{col:<10}"
            f"{next_player_value:+.4f}"
            f"{'':<20}"
            f"{original_player_value:+.4f}"
        )

    print()
    print("=" * 75)
    print("TEST FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    main()
