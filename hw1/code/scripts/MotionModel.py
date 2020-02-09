import sys
import numpy as np
import math
import Utils


class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self, a1, a2, a3, a4):
        # Alphas as parameters
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        x0 = u_t0[0]
        y0 = u_t0[1]
        theta0 = u_t0[2]

        x1 = u_t1[0]
        y1 = u_t1[1]
        theta1 = u_t1[2]

        rot1 = math.atan2(y1 - y0, x1 - x0) - theta0
        trans = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        rot2 = theta1 - theta0 - rot1

        rot1_est = rot1 - \
            np.random.normal(scale=math.sqrt(
                self.a1*rot1*rot1 + self.a2*trans*trans))
        trans_est = trans - np.random.normal(scale=math.sqrt(
            self.a3*trans*trans + self.a4*rot1*rot1 + self.a4*rot2*rot2))
        rot2_est = rot2 - \
            np.random.normal(scale=math.sqrt(
                self.a1*rot2*rot2 + self.a2*trans*trans))

        x = x_t0[0] + trans_est * math.cos(x_t0[2]+rot1_est)
        y = x_t0[1] + trans_est * math.sin(x_t0[2]+rot1_est)
        theta = x_t0[2] + rot1_est + rot2_est
        theta = Utils.trimTheta(theta)
        return np.array([x, y, theta])


if __name__ == "__main__":
    pass
