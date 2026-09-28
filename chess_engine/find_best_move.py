import chess

from minimax import minimax

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