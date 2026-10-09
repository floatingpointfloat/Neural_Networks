import sys, math
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

        # simulation variables:
        # Virtual simulation dimensions so resizing the window doesn't change anything
        self.virtual_width = 1000
        self.virtual_height = 600
        # cart
        self.cart_x = 0
        self.cart_height = 10
        self.cart_width = 25
        # double pendulum
        self.angle_1 = 0.0
        self.angle_2 = 0.0
        self.pendulum_length_1 = 120
        self.pendulum_length_2 = 120

    def set_state(self, cart_x, angle_1, angle_2):
        self.cart_x = cart_x
        self.angle_1 = angle_1
        self.angle_2 = angle_2

        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # background
        painter.fillRect(self.rect(), QColor("#121212"))

        # calculate the scale factor
        scale = min(
            self.width() / self.virtual_width,
            self.height() / self.virtual_height,
        )

        # calculate the offset to center the virtual canvas
        offset_x = (self.width() - self.virtual_width * scale) / 2
        offset_y = (self.height() - self.virtual_height * scale) / 2

        # transform the painter into virtual coordinates
        painter.translate(offset_x, offset_y)
        painter.scale(scale, scale)

        # Virtual simulation area
        sim_area_width = self.virtual_width * 0.9
        sim_area_height = self.virtual_height * 0.9

        sim_area_x = (self.virtual_width - sim_area_width) / 2
        sim_area_y = (self.virtual_height - sim_area_height) / 2

        # drawing boundary
        painter.setPen(QPen(QColor("#008AC5"), 8))
        painter.drawRoundedRect(
            int(sim_area_x),
            int(sim_area_y),
            int(sim_area_width),
            int(sim_area_height),
            20,
            20,
        )

        # drawing the 'rail'
        pen = QPen(QColor("#ffffff"), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(
            int(sim_area_x + sim_area_width / 7),
            int(sim_area_y + sim_area_height / 2),
            int(sim_area_x + sim_area_width - sim_area_width / 7),
            int(sim_area_y + sim_area_height / 2),
        )

        # drawing the cart
        pen = QPen(QColor("#b3b3b3"), 2)
        brush = QBrush(QColor("#8c8c8c"))
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRoundedRect(
            int(sim_area_x + sim_area_width / 2 + self.cart_x - self.cart_width / 2),
            int(sim_area_y + sim_area_height / 2 - self.cart_height / 2),
            int(self.cart_width),  # width
            int(self.cart_height),  # height
            5,
            5,
        )

        # Drawing the double pendulum
        # pivot on the cart
        pivot_x = sim_area_x + sim_area_width / 2 + self.cart_x
        pivot_y = sim_area_y + sim_area_height / 2

        # end of the first pendulum arm
        joint_x = pivot_x + self.pendulum_length_1 * math.sin(self.angle_1)
        joint_y = pivot_y + self.pendulum_length_1 * math.cos(self.angle_1)

        # end of the second pendulum arm
        end_x = joint_x + self.pendulum_length_2 * math.sin(self.angle_2)
        end_y = joint_y + self.pendulum_length_2 * math.cos(self.angle_2)

        # drawing the arms
        pen = QPen(QColor("#ffffff"), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # first arm
        painter.drawLine(
            int(pivot_x),
            int(pivot_y),
            int(joint_x),
            int(joint_y),
        )

        # second arm
        painter.drawLine(
            int(joint_x),
            int(joint_y),
            int(end_x),
            int(end_y),
        )

        # pivt
        pen = QPen(QColor("#dcdcdc"), 2)
        brush = QBrush(QColor("#626262"))
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawEllipse(
            int((sim_area_x + sim_area_width / 2 + self.cart_x) - 3.5),
            int((sim_area_y + sim_area_height / 2) - 3.5),
            7,
            7,
        )

        # joints and masses
        painter.setPen(QPen(QColor("#b3b3b3"), 2))
        painter.setBrush(QBrush(QColor("#1c6b9f")))

        # middle joint
        painter.drawEllipse(
            int(joint_x - 5),
            int(joint_y - 5),
            10,
            10,
        )

        # mass on the end
        painter.drawEllipse(
            int(end_x - 7.5),
            int(end_y - 7.5),
            15,
            15,
        )

        # ending the painter
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

    def set_state(self, cart_x, angle_1, angle_2):
        self.simulation_widget.set_state(cart_x, angle_1, angle_2)


def create_app():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app, window
