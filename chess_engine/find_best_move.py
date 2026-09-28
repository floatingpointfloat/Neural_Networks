import chess
import torch
import time

from board_to_tensor import board_to_tensor

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 1000}
TIME_LIMIT = 30 #in seconds
 
def evaluate_board(board: chess.Board, model, device):
    #evaluates a position using the cnn

    tensor = board_to_tensor(board)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        value = model(tensor).item()

    return value

#check for important remaining moves even if the depth is zero
def quiescence_search(board: chess.Board, alpha, beta, is_maximizing, model, device):
    stand_pat = evaluate_board(board, model, device)

    if is_maximizing:
        if stand_pat >= beta:
            return beta

        if stand_pat > alpha:
            alpha = stand_pat

        for move in order_moves(board):
            if not board.is_capture(move) and not board.gives_check(move):
                continue

            board.push(move)

            evaluation = quiescence_search(board, alpha, beta, False, model, device)

            board.pop()

            if evaluation > alpha:
                alpha = evaluation

            if alpha >= beta:
                break

        return alpha
    else:
        if stand_pat <= alpha:
            return alpha

        if stand_pat < beta:
            beta = stand_pat

        for move in order_moves(board):
            if not board.is_capture(move) and not board.gives_check(move):
                continue

            board.push(move)

            evaluation = quiescence_search(
                board,
                alpha,
                beta,
                True,
                model,
                device
            )

            board.pop()

            if evaluation < beta:
                beta = evaluation

            if beta <= alpha:
                break

        return beta

def order_moves(board:chess.Board):
    moves = list(board.legal_moves)

    def score_moves(move):
        score = 0

        if board.is_capture(move):
            attacker_value = PIECE_VALUES[(board.piece_at(move.from_square)).piece_type]
            if board.is_en_passant(move): #catch en passant squares, no piece would actually be standing on those quares
                victim_value = PIECE_VALUES[chess.PAWN]
            else:
                victim_value = PIECE_VALUES[(board.piece_at(move.to_square)).piece_type]

            score += 100 * (victim_value / attacker_value)

        if move.promotion is not None:
            score += 200

        if board.is_castling(move):
            score += 40

        return score     

    moves.sort(key=score_moves, reverse=True)  
    return moves 

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
        return quiescence_search(board, alpha, beta, is_maximizing, model, device)

    if is_maximizing:
        max_eval = float("-inf")
        for move in order_moves(board):
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

        for move in order_moves(board):
            board.push(move)

            evaluation = minimax(board, depth - 1, alpha, beta, True, model, device)

            board.pop()

            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)

            if beta <= alpha:
                break

        return min_eval

def find_best_move(board: chess.Board, depth, model, device, time_limit=TIME_LIMIT):
    start_time = time.time()

    best_move = None

    if board.turn == chess.WHITE:
        best_value = float("-inf")
    else:
        best_value = float("+inf")

    alpha = float("-inf")
    beta = float("+inf")

    for move in order_moves(board):

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