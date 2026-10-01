import sys

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QApplication, QWidget


class Connect4Board(QWidget):
    move_requested = Signal(int)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Connect Four")
        self.setMinimumSize(700, 600)

        self.board = None
        self.human_turn = True
        self.ai_thinking = False

        self.setMouseTracking(True)
        self.hover_column = -1

    def set_board(self, board):
        self.board = board
        self.update()

    def set_human_turn(self, value):
        self.human_turn = value
        self.update()

    def set_ai_thinking(self, value):
        self.ai_thinking = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Board-Größe
        board_width = min(width * 0.85, 630)
        board_height = board_width * 6 / 7

        x = (width - board_width) / 2
        y = (height - board_height) / 2

        cell_width = board_width / 7
        cell_height = board_height / 6

        # Hintergrund
        painter.fillRect(self.rect(), QColor("#121212"))

        # Board
        painter.setBrush(QBrush(QColor("#1976D2")))
        painter.setPen(Qt.NoPen)

        painter.drawRoundedRect(
            int(x), int(y), int(board_width), int(board_height), 20, 20
        )

        # Spielfeld
        for row in range(6):
            for col in range(7):

                center_x = x + col * cell_width + cell_width / 2
                center_y = y + row * cell_height + cell_height / 2

                radius = min(cell_width, cell_height) * 0.38

                color = QColor("#202020")

                if self.board is not None:

                    # Deine interne Darstellung:
                    #
                    # board[0] = aktueller Spieler
                    # board[1] = Gegner
                    #
                    # Für die GUI interpretieren wir die
                    # aktuelle Perspektive.

                    if self.board[0, row, col] == 1:
                        color = QColor("#F44336")

                    elif self.board[1, row, col] == 1:
                        color = QColor("#FFD600")

                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)

                painter.drawEllipse(
                    int(center_x - radius),
                    int(center_y - radius),
                    int(radius * 2),
                    int(radius * 2),
                )

        # Hover-Anzeige
        if self.hover_column != -1 and self.human_turn and not self.ai_thinking:
            col = self.hover_column

            center_x = x + col * cell_width + cell_width / 2
            center_y = y - cell_height * 0.55

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
        height = self.height()

        board_width = min(width * 0.85, 630)
        board_height = board_width * 6 / 7

        x = (width - board_width) / 2
        y = (height - board_height) / 2

        cell_width = board_width / 7

        mouse_x = event.position().x()
        mouse_y = event.position().y()

        if x <= mouse_x <= x + board_width and y <= mouse_y <= y + board_height:
            self.hover_column = int((mouse_x - x) / cell_width)
        else:
            self.hover_column = -1

        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        if not self.human_turn or self.ai_thinking:
            return

        width = self.width()

        board_width = min(width * 0.85, 630)
        x = (width - board_width) / 2

        cell_width = board_width / 7

        mouse_x = event.position().x()

        if x <= mouse_x <= x + board_width:

            column = int((mouse_x - x) / cell_width)

            if 0 <= column < 7:
                self.move_requested.emit(column)


class GameWindow(Connect4Board):
    pass


def run_gui(board):
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    window = GameWindow()

    window.set_board(board)

    window.resize(800, 700)
    window.show()

    return app, window
