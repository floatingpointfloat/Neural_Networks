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

#minimax search function - alpha beta search tree pruning
def minimax(board: chess.Board, depth, alpha, beta, is_maximizing, model, device):
    if board.is_game_over():
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -1.0
            else:
                return 1.0

        return 0.0

    if depth == 0:
        return evaluate_board(board, model, device)

    if is_maximizing:
        max_eval = float("-inf")
        for move in board.legal_moves:
            board.push(move)

            evaluation = minimax(board, depth - 1, alpha, beta, False, model, device)

            board.pop()

            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)

            if beta <= alpha:
                break

        return max_eval

    else:
        min_eval = float("+inf")

        for move in board.legal_moves:
            board.push(move)

            evaluation = minimax(board, depth - 1, alpha, beta, True, model, device)

            board.pop()

            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)

            if beta <= alpha:
                break

        return min_eval

def find_best_move(board: chess.Board, depth, model, device):
    best_move = None

    if board.turn == chess.WHITE:
        best_value = float("-inf")
    else:
        best_value = float("+inf")

    alpha = float("-inf")
    beta = float("+inf")

    for move in board.legal_moves:

        board.push(move)

        value = minimax(
            board,
            depth - 1,
            alpha,
            beta,
            board.turn == chess.BLACK,
            model,
            device
        )

        board.pop()

        if board.turn == chess.WHITE:

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, value)

        else:

            if value < best_value:
                best_value = value
                best_move = move

            beta = min(beta, value)

    return best_move, best_value