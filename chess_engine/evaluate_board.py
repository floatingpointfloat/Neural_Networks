import chess
import torch

from board_to_tensor import board_to_tensor

def evaluate_board(board: chess.Board, model, device):
    #evaluates a position using the cnn

    tensor = board_to_tensor(board)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        value = model(tensor).item()

    return value