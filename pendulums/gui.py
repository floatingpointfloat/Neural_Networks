import sys

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QPainterPath,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)


class SimulationWidget(QWidget):

    # Signals

    def __init__(self):
        super().__init__()

        # Minimum size of the simulation area
        self.setMinimumSize(800, 600)

        # Receive mouse move events even without a pressed mouse button
        self.setMouseTracking(True)

        # Allow the widget to receive keyboard input
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Timer
        self.simulation_timer = QTimer(self)

    def get_simulation_area_dimensions(self):
        width = self.width()
        height = self.height()

        simulation_area_width = width * 0.9
        simulation_area_height = height * 0.9

        return width, height, simulation_area_width, simulation_area_height

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width, height, simulation_area_width, simulation_area_height = (
            self.get_simulation_area_dimensions()
        )

        # Background
        painter.fillRect(self.rect(), QColor("#121212"))

        # Simulation area
        x = (self.width() - simulation_area_width) / 2
        y = (self.height() - simulation_area_height) / 2

        painter.setPen(QPen(QColor("#444444"), 5))
        painter.drawRoundedRect(int())

        # Center point
        center_x = self.width() / 2
        center_y = self.height() / 2

        painter.setPen(QPen(QColor("#FFFFFF"), 3))
        painter.drawPoint(int(center_x), int(center_y))

        # Vertical debug line
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.drawLine(
            int(center_x),
            int(y),
            int(center_x),
            int(y + simulation_area_height),
        )

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        pass

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def mouseMoveEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event: QKeyEvent):
        pass

    def keyReleaseEvent(self, event: QKeyEvent):
        pass

    def resizeEvent(self, event: QResizeEvent):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pendulum Simulation")
        self.resize(1000, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Simulation widget
        self.simulation_widget = SimulationWidget()

        # Status
        self.status_label = QLabel("Simulation ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFixedHeight(25)

        # Buttons
        self.start_button = QPushButton("Start AI")
        self.reset_button = QPushButton("Reset")

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.reset_button)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.simulation_widget)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)

        central_widget.setLayout(layout)


def create_app():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app, window
