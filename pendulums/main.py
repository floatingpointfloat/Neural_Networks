import sys
import numpy as np

from PySide6.QtCore import QTimer

from gui import create_app
from lagrange import DoublePendulumCart


def main():
    # create app and simulation
    app, window = create_app()
    simulation = DoublePendulumCart(
        cart_friction=0.3,
        joint_friction_1=0.01,
        joint_friction_2=0.01,
        x_min=-2.0,
        x_max=2.0,
    )

    # set simulation parameters
    dt = 1 / 240
    pixels_per_meter = 100
    time = 0.0

    # update simulation
    def update_simulation():
        nonlocal time

        time += dt

        # move cart back and forth
        force = 10.0 * np.sin(2.0 * time)

        simulation.step(dt, force)

        x, theta_1, theta_2 = simulation.get_state()

        # convert meters to pixels
        x_pixels = x * pixels_per_meter

        # update gui
        window.set_state(x_pixels, theta_1, theta_2)

    # create timer
    timer = QTimer()
    timer.timeout.connect(update_simulation)
    window.reset_button.clicked.connect(simulation.reset)
    timer.start(round(1000 / 240))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
