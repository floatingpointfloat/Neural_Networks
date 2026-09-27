import zstandard as zstd
import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
from board_to_tensor import board_to_tensor

RESULT_TO_VALUE = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0
}

class ChessDataset(Dataset):
    def __init__(self, pgn_path, max_games=None, min_elo=None):
        self.positions = []
        self.targets = []

        self.pgn_path = pgn_path
        self.max_games = max_games
        self.min_elo = min_elo

        self.load_games()

    def load_games(self):
        games_loaded = 0