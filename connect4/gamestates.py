"""
Rules and general game state checks
-> for example draw, win, board
"""

import numpy as np

"""
Board structure: 
channel 0: pieces of the currently playing player 
channel1: pieces of the current opponent
channel 2: which player is currently playing, player 1 or 2?
"""
BOARD = np.zeros((3, 6, 7), dtype=np.int8)
PLAYER1 = 0
PLAYER2 = 1


def legal_moves(board):
    free_slots = []
    for column in range(7):
        if board[0, 0, column] == 0 and board[1, 0, column] == 0:
            free_slots.append(column)

    return free_slots


def make_move(board, column):
    if column not in legal_moves(board):
        raise RuntimeError(f"Invalid Move! Can't place a piece on column {column + 1}")

    height = np.sum((board[0, :, column] == 1) | (board[1, :, column] == 1))
    board[0, 5 - height, column] = 1

    return board


def flip_board(board):
    flipped = board.copy()

    flipped[0] = board[1]
    flipped[1] = board[0]
    flipped[2, :, :] = 1 - board[2, :, :]  # opposites player turn now :)

    return flipped


def check_draw(board):
    return not legal_moves(board)


def check_win(board):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    current_player = get_current_player(board)

    for row in range(6):
        for column in range(7):

            if board[0, row, column] == 0 and board[1, row, column] == 0:
                continue

            channel = 0 if board[0, row, column] == 1 else 1

            if channel == 0:
                player = current_player
            else:
                player = 1 - current_player

            for dr, dc in directions:
                count = 1

                r = row + dr
                c = column + dc

                while 0 <= r < 6 and 0 <= c < 7 and board[channel, r, c] == 1:
                    count += 1
                    r += dr
                    c += dc

                r = row - dr
                c = column - dc

                while 0 <= r < 6 and 0 <= c < 7 and board[channel, r, c] == 1:
                    count += 1
                    r -= dr
                    c -= dc

                if count >= 4:
                    return True, player

    return False, None


def is_game_over(board):
    # third channel for draws to differentiate
    win, player = check_win(board)
    if win:
        return True, player, False

    if check_draw(board):
        return True, None, True

    return False, None, False


def get_current_player(board):
    return PLAYER1 if board[2, 0, 0] == 0 else PLAYER2


def copy_board(board):
    return board.copy()


def simulate_move(board, column):
    new_board = board.copy()
    new_board = make_move(new_board, column)

    return flip_board(new_board)

def order_moves(board):
    moves = legal_moves(board)

    def sort_moves(move):
        score = 0
        if move == 3:
            score += 10
        if move == 2 or move == 4:
            score += 7
        if move == 1 or move == 5:
            score += 4
        if move == 0 or move == 6:
            score += 1

        return score

    moves = moves.sort(key=sort_moves, reverse=True)
    return moves

def is_critical_position(board):
    #check if the enemy could win
    for move in legal_moves(board):
        new_board = simulate_move(flip_board(board), move)

        if check_win(new_board):
            return True