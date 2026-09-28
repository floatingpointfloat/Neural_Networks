import torch
import time
from pathlib import Path

from dataset import ChessDataset
from config import DATASET_PATH, PGN_PATH, DATASET_CHECKPOINT_PATH

DATASET_CHECKPOINT_PATH = Path(DATASET_CHECKPOINT_PATH)
DATASET_PATH = Path(DATASET_PATH)

MAX_GAMES = 1000 #processed games

# how many games should be processed before saving a checkpoint
CHECKPOINT_EVERY = 25

MIN_ELO = 2500


# Load the previous checkpoint if one exists
if DATASET_CHECKPOINT_PATH.exists():

    print(
        f"Loading checkpoint: {DATASET_CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        DATASET_CHECKPOINT_PATH,
        map_location="cpu"
    )

    boards = list(checkpoint["boards"])
    targets = list(checkpoint["targets"])
    games_loaded = checkpoint["games_loaded"]

    print(
        f"Resuming from game {games_loaded}"
    )

else:

    print("No checkpoint found. Starting from scratch.")

    boards = []
    targets = []

    games_loaded = 0


# create the dataset
dataset = ChessDataset(
    pgn_path=PGN_PATH,
    max_games=MAX_GAMES,
    min_elo=MIN_ELO,
    start_game=games_loaded
)


start_time = time.time()

new_games = 0
last_game = None


# Load the positions and Stockfish evaluations
for board_tensor, target, game_number in dataset:

    # Detect when a new game starts
    if last_game is None:

        new_games = 1

    elif game_number != last_game:

        new_games += 1

        # The previous game is now completely processed
        # Check if another checkpoint should be create
        if (
            new_games - 1 > 0
            and (new_games - 1) % CHECKPOINT_EVERY == 0
        ):

            print()
            print("Saving checkpoint...")

            boards_tensor = torch.stack(boards)
            targets_tensor = torch.stack(targets)

            torch.save(
                {
                    "boards": boards_tensor,
                    "targets": targets_tensor,
                    "games_loaded": last_game
                },
                DATASET_CHECKPOINT_PATH
            )

            print(
                f"Checkpoint saved after "
                f"{last_game} games."
            )

            print()

    # Add the current position
    boards.append(board_tensor)
    targets.append(target)

    # Remember the current game
    last_game = game_number

    # Print progress every 100 positions
    if len(boards) % 100 == 0:

        elapsed = time.time() - start_time

        print(
            f"Positions: {len(boards)} | "
            f"New games: {new_games} | "
            f"Time: {round(elapsed, 2)}s"
        )

# Convert the lists into tensors
boards = torch.stack(boards)
targets = torch.stack(targets)


# Save the finished dataset
torch.save(
    {
        "boards": boards,
        "targets": targets
    },
    DATASET_PATH
)


print()
print("Dataset saved!")
print(f"Boards:  {boards.shape}")
print(f"Targets: {targets.shape}")
print(f"Path:    {DATASET_PATH}")