import torch

from dataset import ChessDataset
from config import DATA_PATH, PGN_PATH

MAX_GAMES = 5

# create the dataset
dataset = ChessDataset(
    pgn_path=PGN_PATH,
    max_games=MAX_GAMES,
    min_elo=2000
)


boards = []
targets = []


# load the positions and Stockfish evaluations
for board_tensor, target in dataset:

    boards.append(board_tensor)
    targets.append(target)


# convert the lists into tensors
boards = torch.stack(boards)
targets = torch.stack(targets)


# save the dataset
torch.save(
    {
        "boards": boards,
        "targets": targets
    },
    DATA_PATH
)


print()
print("Dataset saved!")
print(f"Boards:  {boards.shape}")
print(f"Targets: {targets.shape}")
print(f"Path:    {DATA_PATH}")