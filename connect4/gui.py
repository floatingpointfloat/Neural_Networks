from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from gamestates import get_current_player


class Connect4Board(QWidget):
    move_requested = Signal(int)

    def __init__(self):
        super().__init__
        self.setMinimumSize(700, 600)

        # gamestate
        self.board = None

        self.human_turn = True
        self.ai_turn = False

        self.hover_column = -1  # the column that the player is currently selecting

        self.setMouseTracking(True)

    def set_board(self, board):
        self.board = board
        self.update()

    def set_human_turn(self, value):
        self.human_turn = value
        self.update()

    def set_ai_turn(self, value):
        self.ai_turn = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(QPainter.Antialiasing, True)

        width = self.width()
        height = self.height()

        board_width = min(width * 0.85, 630)
        board_height = board_width * 6 / 7

        board_x = (width - board_width) / 2
        board_y = (height - board_height) / 2

        cell_width = board_width / 7
        cell_height = board_height / 6

        # fill the background
        painter.fillRect(self.rect(), QColor("#121212"))

        # paint the board
        painter.setBrush(QBrush(QColor("#1976D2")))

        painter.setPen(Qt.NoPen)

        painter.drawRoundedRect(
            int(board_x), int(board_y), int(board_width), int(board_height), 20, 20
        )

        # paint the pieces and the holes and so on
        for row in range(6):
            for column in range(7):
                center_x = board_x + column * cell_width + cell_width / 2
                center_y = board_y + row * cell_height + cell_height / 2

                radius = min(cell_width, cell_height) * 0.38

                # default color
                color = QColor("#202020")

                if self.board is not None:
                    if get_current_player(self.board) == 1 and (
                        self.board[0, row, column] == 1
                        or self.board[1, row, column] == 1
                    ):
                        color = QColor("#F44336")
                    elif get_current_player(self.board) == 0 and (
                        self.board[0, row, column] == 1
                        or self.board[1, row, column] == 1
                    ):
                        color = QColor("#100CFF")

                painter.setBrush(QBrush(color))

                painter.drawEllipse(
                    int(center_x - radius),
                    int(center_y - radius),
                    int(radius * 2),
                    int(radius * 2),
                )

        # hover piece :)
        if self.hover_column != -1 and self.human_turn and not self.ai_thinking:
            column = self.hover_column

            center_x = board_x + column * cell_width + cell_width / 2

            center_y = board_y - cell_height * 0.55

            radius = min(cell_width, cell_height) * 0.25

            painter.setBrush(QBrush(QColor(244, 67, 54, 150)))

            painter.drawEllipse(
                int(center_x - radius),
                int(center_y - radius),
                int(radius * 2),
                int(radius * 2),
            )

        painter.end()

    def mouseMoveEvent(self, event):
        width = self.width()
        board_width = min(width * 0.85, 630)
        board_x = (width - board_width) / 2
        cell_width = board_width / 7
        mouse_x = event.position().x()

        # is the mouse on the board?
        if board_x <= mouse_x <= board_x + board_width:
            self.hover_column = int((mouse_x - board_x) / cell_width)
        else:
            self.hover_column = -1

        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        if self.ai_turn:
            return
        elif not self.human_turn:
            return

        width = self.width()
        board_width = min(width * 0.85, 630)
        board_x = (width - board_width) / 2
        cell_width = board_width / 7
        mouse_x = event.position().x()

        if board_x <= mouse_x <= board_x + board_width:
            column = int((mouse_x - board_x) / cell_width)
            if 0 <= column < 7:
                # "sends" a message to play.py
                self.move_requested.emit(column)
