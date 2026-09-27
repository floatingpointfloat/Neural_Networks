from model import model
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChessDataset
from config import PGN_PATH

ALLOW_TRAINING = True

if ALLOW_TRAINING:
    pass
else:
    raise "No Training currently allowed. Please do not train the model"
    exit()


dataset = ChessDataset(
    pgn_path=PGN_PATH,
    max_games=100
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)


for boards, targets in loader:

    print(boards.shape)
    print(targets.shape)

    break