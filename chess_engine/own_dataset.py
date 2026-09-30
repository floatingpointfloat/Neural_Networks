import io
import zstandard as zstd
import chess
import chess.engine
import chess.pgn
import torch
from torch.utils.data import IterableDataset

from board_to_tensor import board_to_tensor
from config import STOCKFISH_PATH

ANALYZE_EVERY_NTH_MOVE = 3
STOCKFISH_TIME = 1


class ChessDataset(IterableDataset):

    def __init__(self, pgn_path, max_games=None, min_elo=2000, start_game=0):

        self.pgn_path = pgn_path
        self.max_games = max_games
        self.min_elo = min_elo

        self.start_game = start_game

    def __iter__(self):

        games_loaded = 0

        new_games = 0

        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)  # start stockfish

        try:
            with open(self.pgn_path, "rb") as lichess_games:

                dctx = zstd.ZstdDecompressor()

                # loading the compressed lichess games over zstd
                with dctx.stream_reader(lichess_games) as reader:

                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                    while True:

                        game = chess.pgn.read_game(text_stream)

                        if game is None:
                            break

                        headers = game.headers

                        # Only use games with the min_elo requirement
                        if self.min_elo is not None:

                            try:
                                white_elo = int(headers["WhiteElo"])
                                black_elo = int(headers["BlackElo"])

                            except (KeyError, ValueError):
                                continue

                            if white_elo < self.min_elo or black_elo < self.min_elo:
                                continue

                        games_loaded += 1

                        # skip already processed games
                        if games_loaded <= self.start_game:
                            continue

                        new_games += 1

                        # Break the loop after reaching the maximum number of loaded games
                        if self.max_games is not None and new_games > self.max_games:
                            break

                        board = game.board()

                        # one training example per position
                        for i, move in enumerate(game.mainline_moves()):
                            if i % ANALYZE_EVERY_NTH_MOVE == 0:
                                tensor = board_to_tensor(board)

                                analysis = engine.analyse(
                                    board, chess.engine.Limit(time=STOCKFISH_TIME)
                                )

                                score = (
                                    analysis["score"].white().score(mate_score=10000)
                                )

                                target = torch.tanh(
                                    torch.tensor(score / 400.0, dtype=torch.float32)
                                )

                                # print(
                                # f"Stockfish: {score:>6} cp | "
                                # f"Target: {target.item():+.4f}"
                                # )

                                yield (tensor, target, games_loaded)

                            board.push(move)

                        if games_loaded % 500 == 0:
                            print(f"Loaded {games_loaded} games")
        finally:
            # close stockfish
            engine.quit()
