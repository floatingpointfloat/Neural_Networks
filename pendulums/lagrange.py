import numpy as np


class DoublePendulumCart:
    def __init__(
        self,
        cart_mass=1.0,
        mass_1=0.2,
        mass_2=0.2,
        length_1=0.5,
        length_2=0.5,
        gravity=9.81,
        cart_friction=0.15,
        joint_friction_1=0.02,
        joint_friction_2=0.02,
        x_min=-2.0,
        x_max=2.0,
    ):
        # set physical parameters
        self.M = cart_mass
        self.m1 = mass_1
        self.m2 = mass_2

        self.l1 = length_1
        self.l2 = length_2
        self.g = gravity

        # set friction coefficients
        self.cart_friction = cart_friction
        self.joint_friction_1 = joint_friction_1
        self.joint_friction_2 = joint_friction_2

        # set rail limits
        self.x_min = x_min
        self.x_max = x_max

        if self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max")

        # initial state
        self.x = 0.0
        self.theta_1 = np.deg2rad(170)
        self.theta_2 = np.deg2rad(175)

        self.x_dot = 0.0
        self.theta_1_dot = 0.0
        self.theta_2_dot = 0.0

    def calculate_accelerations(self, force=0.0):
        # get physical parameters
        M = self.M
        m1 = self.m1
        m2 = self.m2

        l1 = self.l1
        l2 = self.l2
        g = self.g

        # calculate trigonometric values
        theta_1 = self.theta_1
        theta_2 = self.theta_2

        cos_1 = np.cos(theta_1)
        cos_2 = np.cos(theta_2)

        sin_1 = np.sin(theta_1)
        sin_2 = np.sin(theta_2)

        cos_12 = np.cos(theta_1 - theta_2)
        sin_12 = np.sin(theta_1 - theta_2)

        # build mass matrix
        mass_matrix = np.array(
            [
                [
                    M + m1 + m2,
                    (m1 + m2) * l1 * cos_1,
                    m2 * l2 * cos_2,
                ],
                [
                    (m1 + m2) * l1 * cos_1,
                    (m1 + m2) * l1**2,
                    m2 * l1 * l2 * cos_12,
                ],
                [
                    m2 * l2 * cos_2,
                    m2 * l1 * l2 * cos_12,
                    m2 * l2**2,
                ],
            ],
            dtype=float,
        )

        # calculate movement-dependent terms
        C = np.array(
            [
                -(m1 + m2) * l1 * sin_1 * self.theta_1_dot**2
                - m2 * l2 * sin_2 * self.theta_2_dot**2,
                m2 * l1 * l2 * sin_12 * self.theta_2_dot**2,
                -m2 * l1 * l2 * sin_12 * self.theta_1_dot**2,
            ],
            dtype=float,
        )

        # calculate gravity terms
        G = np.array(
            [
                0.0,
                (m1 + m2) * g * l1 * sin_1,
                m2 * g * l2 * sin_2,
            ],
            dtype=float,
        )

        # calculate external forces and friction
        Q = np.array(
            [
                force - self.cart_friction * self.x_dot,
                -self.joint_friction_1 * self.theta_1_dot,
                -self.joint_friction_2 * self.theta_2_dot,
            ],
            dtype=float,
        )

        # solve the equations of motion
        accelerations = np.linalg.solve(mass_matrix, Q - C - G)

        return accelerations

    def step(self, dt, force=0.0):
        if dt <= 0:
            raise ValueError("dt must be greater than zero")

        # calculate accelerations
        x_ddot, theta_1_ddot, theta_2_ddot = self.calculate_accelerations(force)

        # update velocities
        self.x_dot += x_ddot * dt
        self.theta_1_dot += theta_1_ddot * dt
        self.theta_2_dot += theta_2_ddot * dt

        # update positions and angles
        self.x += self.x_dot * dt
        self.theta_1 += self.theta_1_dot * dt
        self.theta_2 += self.theta_2_dot * dt

        # keep the cart on the rail
        if self.x >= self.x_max:
            self.x = self.x_max

            if self.x_dot > 0:
                self.x_dot = 0.0

        elif self.x <= self.x_min:
            self.x = self.x_min

            if self.x_dot < 0:
                self.x_dot = 0.0

    def get_state(self):
        return self.x, self.theta_1, self.theta_2
