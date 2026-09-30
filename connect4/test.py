import torch
import numpy as np

from load_model import load_model


def make_board(player1, player2, turn):
    """
    player1/player2: Liste von (row, column)
    turn: 0 = Player 1, 1 = Player 2
    """

    board = np.zeros((3, 6, 7), dtype=np.float32)

    for row, col in player1:
        board[0, row, col] = 1.0

    for row, col in player2:
        board[1, row, col] = 1.0

    board[2, :, :] = turn

    return torch.tensor(board, dtype=torch.float32)


def predict(model, board, device):
    board = board.unsqueeze(0).to(device)

    with torch.no_grad():
        value = model(board)

    return value.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model()

    # --------------------------------------------------
    # TEST 1: Leeres Board
    # --------------------------------------------------

    board = make_board(player1=[], player2=[], turn=0)

    value = predict(model, board, device)

    print(f"Empty board:       {value:+.4f}")

    # --------------------------------------------------
    # TEST 2: Player 1 hat bereits vier in einer Reihe
    # --------------------------------------------------

    board = make_board(
        player1=[
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
        ],
        player2=[],
        turn=1,
    )

    value = predict(model, board, device)

    print(f"Player 1 wins:      {value:+.4f}")

    # --------------------------------------------------
    # TEST 3: Player 2 hat bereits vier in einer Reihe
    # --------------------------------------------------

    board = make_board(
        player1=[],
        player2=[
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
        ],
        turn=0,
    )

    value = predict(model, board, device)

    print(f"Player 2 wins:      {value:+.4f}")

    # --------------------------------------------------
    # TEST 4: Player 1 hat unmittelbar eine Gewinnchance
    # --------------------------------------------------

    board = make_board(
        player1=[
            (5, 0),
            (5, 1),
            (5, 2),
        ],
        player2=[],
        turn=0,
    )

    value = predict(model, board, device)

    print(f"Player 1 threat:    {value:+.4f}")


if __name__ == "__main__":
    main()
