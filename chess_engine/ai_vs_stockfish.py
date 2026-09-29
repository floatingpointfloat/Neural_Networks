import io
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch
import zstandard as zstd

from config import PGN_PATH, STOCKFISH_PATH
from model import ChessValueNet
from board_to_tensor import board_to_tensor
from find_best_move import find_best_move


# ============================================================
# Einstellungen
# ============================================================

NUMBER_OF_POSITIONS = 10

# Deine KI
DEPTH = 3
print(f"Depth searched: {DEPTH}")

# Stockfish
STOCKFISH_TIME = 1.0

# Nur Spiele mit ausreichend hoher Elo verwenden
MIN_ELO = 2000

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "best.pt"
)


# ============================================================
# Gerät
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Modell laden
# ============================================================

print()
print("Loading model...")

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
# Direkte Bewertung einer Stellung durch das Netzwerk
# ============================================================

def evaluate_position(board):

    tensor = board_to_tensor(board)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        value = model(
            tensor
        ).item()

    return value


# ============================================================
# Zufällige Schachpositionen aus PGN laden
# ============================================================

def get_positions(number_of_positions):

    positions = []

    print()
    print("Reading PGN...")

    with open(
        PGN_PATH,
        "rb"
    ) as pgn_file:

        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(
            pgn_file
        ) as reader:

            text_stream = io.TextIOWrapper(
                reader,
                encoding="utf-8"
            )

            while len(positions) < number_of_positions:

                game = chess.pgn.read_game(
                    text_stream
                )

                if game is None:
                    break

                headers = game.headers

                # ------------------------------------------------
                # Elo filtern
                # ------------------------------------------------

                try:

                    white_elo = int(
                        headers["WhiteElo"]
                    )

                    black_elo = int(
                        headers["BlackElo"]
                    )

                except (
                    KeyError,
                    ValueError
                ):

                    continue

                if (
                    white_elo < MIN_ELO
                    or black_elo < MIN_ELO
                ):

                    continue

                # ------------------------------------------------
                # Alle Positionen des Spiels sammeln
                # ------------------------------------------------

                board = game.board()

                game_positions = []

                for move in game.mainline_moves():

                    # Position VOR dem Zug speichern
                    if not board.is_game_over():

                        game_positions.append(
                            board.fen()
                        )

                    board.push(move)

                if not game_positions:
                    continue

                # ------------------------------------------------
                # Zufällige Position aus diesem Spiel
                # ------------------------------------------------

                fen = random.choice(
                    game_positions
                )

                positions.append(
                    fen
                )

                print(
                    f"Loaded position "
                    f"{len(positions)}/"
                    f"{number_of_positions}"
                )

    return positions


# ============================================================
# Stockfish
# ============================================================

def start_stockfish():

    print()
    print("Starting Stockfish...")
    print("Path:", STOCKFISH_PATH)

    engine = chess.engine.SimpleEngine.popen_uci(
        STOCKFISH_PATH
    )

    # Nur einen Stockfish-Thread verwenden.
    engine.configure({
        "Threads": 1,
        "Hash": 128
    })

    print("Stockfish loaded.")

    return engine


# ============================================================
# Stockfish Analyse
# ============================================================

def analyze_stockfish(
    engine,
    board
):

    result = engine.analyse(
        board,
        chess.engine.Limit(
            time=STOCKFISH_TIME
        )
    )

    best_move = result["pv"][0]

    score = result["score"].pov(
        board.turn
    )

    score_cp = score.score(
        mate_score=10000
    )

    return best_move, score_cp


# ============================================================
# Stockfish Bewertung eines bestimmten Zuges
# ============================================================

def evaluate_move(
    engine,
    board,
    move
):

    test_board = board.copy()

    test_board.push(move)

    result = engine.analyse(
        test_board,
        chess.engine.Limit(
            time=STOCKFISH_TIME
        )
    )

    # Bewertung wieder aus Sicht des Spielers,
    # der den Zug gemacht hat.
    score = result["score"].pov(
        board.turn
    )

    return score.score(
        mate_score=10000
    )


# ============================================================
# Hauptprogramm
# ============================================================

def main():

    positions = get_positions(
        NUMBER_OF_POSITIONS
    )

    if not positions:

        print()
        print("Keine Positionen gefunden.")
        return

    engine = None

    try:

        engine = start_stockfish()

        same_move_count = 0

        total_cpl = 0

        successful_tests = 0

        # --------------------------------------------------------
        # Zusätzliche Statistiken für das neuronale Netzwerk
        # --------------------------------------------------------

        total_nn_before = 0.0
        total_nn_after = 0.0

        print()
        print("=" * 60)
        print("STARTING TEST")
        print("=" * 60)

        for index, fen in enumerate(
            positions,
            start=1
        ):

            print()
            print("=" * 60)
            print(
                f"Position {index}/"
                f"{len(positions)}"
            )
            print("=" * 60)

            board = chess.Board(fen)

            print()
            print(board)
            print()

            print(
                "FEN:",
                fen
            )

            print(
                "Side to move:",
                "White"
                if board.turn == chess.WHITE
                else "Black"
            )

            # ====================================================
            # Direkte NN-Bewertung der Ausgangsstellung
            # ====================================================

            print()
            print(
                "Neural network is "
                "evaluating position..."
            )

            nn_before = evaluate_position(
                board
            )

            print(
                f"NN evaluation before move: "
                f"{nn_before:+.6f}"
            )

            # ====================================================
            # KI
            # ====================================================

            print()
            print("Your AI is thinking...")

            ai_move, ai_eval = find_best_move(
                board,
                DEPTH,
                model,
                device
            )

            if ai_move is None:

                print(
                    "AI has no legal move."
                )

                continue

            ai_san = board.san(
                ai_move
            )

            print(
                "AI move:",
                ai_san
            )

            print(
                "AI UCI:",
                ai_move
            )

            print(
                f"AI search evaluation: "
                f"{ai_eval:+.6f}"
            )

            # ====================================================
            # Direkte NN-Bewertung NACH dem AI-Zug
            # ====================================================

            ai_board = board.copy()

            ai_board.push(
                ai_move
            )

            nn_after = evaluate_position(
                ai_board
            )

            print(
                f"NN evaluation after AI move: "
                f"{nn_after:+.6f}"
            )

            # ====================================================
            # Stockfish
            # ====================================================

            print()
            print(
                "Stockfish is thinking..."
            )

            try:

                sf_move, sf_score = (
                    analyze_stockfish(
                        engine,
                        board
                    )
                )

            except Exception as error:

                print()
                print(
                    "Stockfish failed:"
                )

                print(
                    repr(error)
                )

                print()
                print(
                    "Skipping this position..."
                )

                continue

            sf_san = board.san(
                sf_move
            )

            print(
                "Stockfish move:",
                sf_san
            )

            print(
                "Stockfish UCI:",
                sf_move
            )

            print(
                f"Stockfish evaluation: "
                f"{sf_score / 100:+.2f}"
            )

            # ====================================================
            # Vergleich der Züge
            # ====================================================

            same_move = (
                ai_move == sf_move
            )

            if same_move:

                same_move_count += 1

                print()
                print(
                    "MATCH: AI found the "
                    "same move as Stockfish!"
                )

            else:

                print()
                print(
                    "DIFFERENT MOVE"
                )

            # ====================================================
            # Stockfish bewertet KI-Zug
            # ====================================================

            print()
            print(
                "Evaluating AI move "
                "with Stockfish..."
            )

            try:

                ai_move_score = (
                    evaluate_move(
                        engine,
                        board,
                        ai_move
                    )
                )

            except Exception as error:

                print(
                    "Could not evaluate "
                    "AI move:"
                )

                print(
                    repr(error)
                )

                continue

            print(
                f"Stockfish evaluation "
                f"of AI move: "
                f"{ai_move_score / 100:+.2f}"
            )

            # ====================================================
            # Centipawn Loss
            # ====================================================

            cpl = (
                sf_score
                - ai_move_score
            )

            # Keine negativen Werte
            cpl = max(
                0,
                cpl
            )

            total_cpl += cpl

            # NN-Statistiken
            total_nn_before += nn_before
            total_nn_after += nn_after

            successful_tests += 1

            print(
                f"Centipawn loss: "
                f"{cpl:.1f}"
            )

            # ====================================================
            # Zusammenfassung dieser Position
            # ====================================================

            print()
            print(
                "-" * 60
            )

            print(
                "POSITION SUMMARY"
            )

            print(
                f"NN before move:       "
                f"{nn_before:+.6f}"
            )

            print(
                f"NN search evaluation: "
                f"{ai_eval:+.6f}"
            )

            print(
                f"NN after AI move:     "
                f"{nn_after:+.6f}"
            )

            print(
                f"Stockfish best move:  "
                f"{sf_score / 100:+.2f}"
            )

            print(
                f"Stockfish AI move:    "
                f"{ai_move_score / 100:+.2f}"
            )

            print(
                f"CPL:                  "
                f"{cpl:.1f}"
            )

        # ========================================================
        # Gesamtergebnis
        # ========================================================

        print()
        print()
        print("=" * 60)
        print("RESULT")
        print("=" * 60)

        print(
            "Positions tested:",
            successful_tests
        )

        if successful_tests > 0:

            percentage = (
                same_move_count
                / successful_tests
                * 100
            )

            average_cpl = (
                total_cpl
                / successful_tests
            )

            average_nn_before = (
                total_nn_before
                / successful_tests
            )

            average_nn_after = (
                total_nn_after
                / successful_tests
            )

            print()
            print(
                "Same move as Stockfish:",
                f"{same_move_count}"
                f"/{successful_tests}"
            )

            print(
                "Same move percentage:",
                f"{percentage:.1f}%"
            )

            print(
                "Average centipawn loss:",
                f"{average_cpl:.1f}"
            )

            print()
            print(
                "Average NN evaluation "
                "before move:",
                f"{average_nn_before:+.6f}"
            )

            print(
                "Average NN evaluation "
                "after AI move:",
                f"{average_nn_after:+.6f}"
            )

        else:

            print(
                "No successful Stockfish "
                "analyses."
            )

    finally:

        if engine is not None:

            print()
            print(
                "Closing Stockfish..."
            )

            try:

                engine.quit()

            except Exception:

                pass

            print(
                "Stockfish closed."
            )


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    main()