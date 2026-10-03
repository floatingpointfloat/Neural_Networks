from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

from gamestates import get_current_player


class Connect4Board(QWidget):

    move_requested = Signal(int)

    def __init__(self):
        super().__init__()

        self.setMinimumSize(700, 600)

        self.board = None
        self.last_move = None

        self.human_turn = True
        self.ai_turn = False

        self.hover_column = -1

        self.setMouseTracking(True)

    def set_board(self, board, last_move=None):
        self.board = board
        self.last_move = last_move
        self.update()

    def set_human_turn(self, value):
        self.human_turn = value
        self.update()

    def set_ai_turn(self, value):
        self.ai_turn = value
        self.update()

    def get_piece_color(self, row, column):
        if self.board is None:
            return QColor("#202020")

        current_player = get_current_player(self.board)

        if self.board[0, row, column] == 1:
            player = current_player

        elif self.board[1, row, column] == 1:
            player = 1 - current_player

        else:
            return QColor("#202020")

        if player == 0:
            return QColor("#100CFF")

        return QColor("#F44336")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        width = self.width()
        height = self.height()

        board_width = width * 0.7
        board_height = board_width * 6 / 7

        board_x = (width - board_width) / 2
        board_y = (height - board_height) * 0.75

        cell_width = board_width / 7
        cell_height = board_height / 6

        painter.fillRect(self.rect(), QColor("#121212"))

        painter.setBrush(QBrush(QColor("#1976D2")))
        painter.setPen(Qt.NoPen)

        painter.drawRoundedRect(
            int(board_x), int(board_y), int(board_width), int(board_height), 20, 20
        )

        for row in range(6):
            for column in range(7):

                center_x = board_x + column * cell_width + cell_width / 2

                center_y = board_y + row * cell_height + cell_height / 2

                radius = min(cell_width, cell_height) * 0.38

                color = self.get_piece_color(row, column)

                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)

                painter.drawEllipse(
                    int(center_x - radius),
                    int(center_y - radius),
                    int(radius * 2),
                    int(radius * 2),
                )

                # Mark the last move
                if self.last_move == (row, column):

                    painter.setBrush(Qt.NoBrush)

                    painter.setPen(QPen(QColor(255, 255, 255, 100), 4))

                    painter.drawEllipse(
                        int(center_x - radius - 2),
                        int(center_y - radius - 2),
                        int(radius * 2 + 4),
                        int(radius * 2 + 4),
                    )

        # Hover piece
        if self.hover_column != -1 and self.human_turn and not self.ai_turn:
            column = self.hover_column
            radius = min(cell_width, cell_height) * 0.25

            center_x = board_x + column * cell_width + cell_width / 2
            center_y = board_y - radius - 10

            current_player = get_current_player(self.board)

            if current_player == 0:
                color = QColor(16, 12, 255, 150)
            else:
                color = QColor(244, 67, 54, 150)

            painter.setBrush(QBrush(color))

            painter.setPen(Qt.NoPen)

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

        if not self.human_turn:
            return

        width = self.width()

        board_width = min(width * 0.85, 630)
        board_x = (width - board_width) / 2
        cell_width = board_width / 7

        mouse_x = event.position().x()

        if board_x <= mouse_x <= board_x + board_width:

            column = int((mouse_x - board_x) / cell_width)

            if 0 <= column < 7:
                self.move_requested.emit(column)


class GameWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Connect-4")
        self.resize(900, 750)

        self.board_widget = Connect4Board()

        self.status_label = QLabel("Your turn")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(25)

        self.new_game_button = QPushButton("New Game")

        layout = QVBoxLayout()

        layout.addWidget(self.board_widget)
        layout.addWidget(self.status_label)
        layout.addWidget(self.new_game_button)

        self.setLayout(layout)

    def set_status(self, text):
        self.status_label.setText(text)


def create_app():
    app = QApplication([])

    window = GameWindow()

    window.show()

    return app, window
