import chess
from evaluate_board import evaluate_board

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