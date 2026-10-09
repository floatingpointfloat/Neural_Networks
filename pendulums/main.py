import sys

from PySide6.QtCore import QTimer

from gui import create_app
from lagrange import DoublePendulumCart


def main():
    # create app and simulation
    app, window = create_app()
    simulation = DoublePendulumCart()

    # set simulation parameters
    dt = 1 / 240
    force = 1.0
    pixels_per_meter = 100

    # update simulation
    def update_simulation():
        simulation.step(dt, force)

        x, theta_1, theta_2 = simulation.get_state()

        # convert meters to pixels
        x_pixels = x * pixels_per_meter

        # update gui
        window.set_state(x_pixels, theta_1, theta_2)

    # create timer
    timer = QTimer()
    timer.timeout.connect(update_simulation)
    timer.start(round(1000 / 240))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
