import torch
import chess
from pathlib import Path

from datasets import load_dataset

from board_to_tensor import board_to_tensor
from config import DATASET_CHECKPOINT_PATH, DATASET_PATH

DATASET_CHECKPOINT_PATH = Path(DATASET_CHECKPOINT_PATH)
DATASET_PATH = Path(DATASET_PATH)

NUM_POSITIONS = 250_000
MIN_DEPTH = 18

def score_to_target(score): #make a readable target out of the score
    target = torch.tanh(torch.tensor(score / 400.0, dtype=torch.float32))
    return target

#loading the lichess dataset
print("Loading the Lichess dataset...")

dataset = load_dataset(
    "Lichess/chess-position-evaluations",
    split="train",
    streaming=True
)

print("Dataset loaded")
print(f"Goal: {NUM_POSITIONS} positions")
print(f"Minimum stockfish depth: {MIN_DEPTH}")
print()

#dataset parameters
boards = []
targets = []

checked = 0
skipped = 0
duplicates = 0
seen_fens = set()

#enumerating the dataset
for position in dataset:
    checked += 1

    fen = position.get("fen")
    if not fen:
        skipped += 1
        continue

    if fen in seen_fens:
        duplicates += 1
        continue
    seen_fens.add(fen)

    depth = position.get("depth")
    if depth == None or (depth < MIN_DEPTH):
        skipped += 1
        continue

    score = position.get("cp")
    if score is None:
        skipped += 1
        continue

    #fen to python chess board
    try:
        board = chess.Board(fen)
    except ValueError:
        skipped += 1
        continue

    #board to tensor
    tensor = board_to_tensor(board)

    #target score
    target = score_to_target(score)

    #saving the data
    boards.append(tensor)
    targets.append(target) 

    #show progress
    if len(boards) % 10 == 0:

        print(
            f"{len(boards):,} / "
            f"{NUM_POSITIONS:,} Positionen"
        )

    #breaking out of the loop after enough positions
    if len(boards) >= NUM_POSITIONS:
        break

#making tensors out of the lists
boards = torch.stack(boards)
targets = torch.stack(targets)

#creating the dataset
dataset_dict = {"boards": boards,
                "targets": targets}

#saving
torch.save(dataset_dict,
           DATASET_PATH)

#debug information
print()
print(f"Saved dataset at path {DATASET_PATH}")
print(f"Boards: {boards.shape}")
print(f"Targets: {targets.shape}")
print()
print("Target statistics:")
print(f"Mean: {targets.mean().item():.6f}")
print(f"Std:  {targets.std().item():.6f}")
print(f"Min:  {targets.min().item():.6f}")
print(f"Max:  {targets.max().item():.6f}")
print()
print(f"Checked entries: {checked}")
print(f"Skipped entries: {skipped}")
print(f"Duplicates:      {duplicates:,}")