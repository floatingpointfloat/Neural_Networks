import torch
import chess
import numpy as np

#shaping the board to be recognised by the neural network
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    #matrix in form of the input channels of the neural network
    matrix = np.zeros((18, 8, 8), dtype=np.float32)

    piece_indices = {chess.PAWN: 0, 
                    chess.KNIGHT: 1, 
                    chess.BISHOP: 2, 
                    chess.ROOK: 3, 
                    chess.QUEEN: 4, 
                    chess.KING: 5
                    }

    #all the positions on the board, channel 1 - 12
    for square, piece in board.piece_map().items():
        piece_type = piece_indices[piece.piece_type]

        #white: channels 0 - 5
        #black: channels 6 - 11
        if piece.color == chess.WHITE:
            channel = piece_type
        else:
            channel = piece_type + 6

        row = 7 - chess.square_rank(square)
        column = chess.square_file(square)

        matrix[channel, row, column] = 1.0

    # channel 13 - side to move
    if board.turn == chess.WHITE:
        matrix[12, :, :] = 1.0

    #channel 14 - white kingside casteling
    if board.has_kingside_castling_rights(chess.WHITE):
        matrix[13, :, :] = 1.0

    #channel 15 - white queenside casteling
    if board.has_queenside_castling_rights(chess.WHITE):
        matrix[14, :, :] = 1.0

    #channel 16 - black kingside casteling
    if board.has_kingside_castling_rights(chess.BLACK):
        matrix[15, :, :] = 1.0

    #channel 17 - black queenside casteling
    if board.has_queenside_castling_rights(chess.BLACK):
        matrix[16, :, :] = 1.0

    #channel 18 - en passant square :) 
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        column = chess.square_file(board.ep_square)

        matrix[17, row, column] = 1.0

    return torch.from_numpy(matrix)