import io
import zstandard as zstd
import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset

from board_to_tensor import board_to_tensor


RESULT_TO_VALUE = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0
}


class ChessDataset(IterableDataset):

    def __init__(self, pgn_path, max_games=None, min_elo=2000):

        self.pgn_path = pgn_path
        self.max_games = max_games
        self.min_elo = min_elo

    def __iter__(self):

        games_loaded = 0

        with open(self.pgn_path, "rb") as lichess_games:

            dctx = zstd.ZstdDecompressor()

            # loading the compressed lichess games over zstd
            with dctx.stream_reader(lichess_games) as reader:

                text_stream = io.TextIOWrapper(
                    reader,
                    encoding="utf-8"
                )

                while True:

                    game = chess.pgn.read_game(text_stream)

                    if game is None:
                        break

                    # Break the loop after reaching the maximum number of loaded games
                    if (
                        self.max_games is not None
                        and games_loaded >= self.max_games
                    ):
                        break

                    headers = game.headers
                    result = headers.get("Result")

                    # Ignoring games without a valid result
                    if result not in RESULT_TO_VALUE:
                        continue

                    # Only use games with the min_elo requirement
                    if self.min_elo is not None:

                        try:
                            white_elo = int(headers["WhiteElo"])
                            black_elo = int(headers["BlackElo"])

                        except:
                            continue

                        if (
                            white_elo < self.min_elo
                            or black_elo < self.min_elo
                        ):
                            continue

                    target = RESULT_TO_VALUE[result]
                    board = game.board()

                    # one training example per position
                    for move in game.mainline_moves():

                        tensor = board_to_tensor(board)

                        # Yield the position directly to the DataLoader
                        # instead of storing all positions in RAM
                        yield (
                            tensor,
                            torch.tensor(
                                target,
                                dtype=torch.float32
                            )
                        )

                        board.push(move)

                    games_loaded += 1

                    if games_loaded % 100 == 0:
                        print(
                            f"Loaded {games_loaded} games"
                        )