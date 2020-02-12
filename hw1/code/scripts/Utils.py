import numpy as np
from numpy import cos, sin
from scipy.integrate import quad
import scipy.stats
import math
# integrates the gaussian distribution from x1 to x2


def integrateGaussian(mean, std, x1, x2):
    # def normal_distribution_function(x):
    #     return scipy.stats.norm.pdf(x, mean, std)

    # res, err = quad(normal_distribution_function, x1, x2)
    return scipy.stats.norm.cdf(x2, mean, std) - scipy.stats.norm.pdf(x1, mean, std)
    return res


def calcGaussian(mean, std, x):
    return scipy.stats.norm(mean, std).pdf(x)


def transformRobotToWorld(x, y, theta, x_robot_in_world, y_robot_in_world):
    xy = np.array([x, y]).reshape(2, 1)
    trans = np.array([x_robot_in_world, y_robot_in_world]).reshape(2, 1)
    R = np.array([[cos(theta), -sin(theta)],
                  [sin(theta), cos(theta)]])
    return np.dot(R, xy) + trans


def trimTheta(theta):
    trimmed = theta
    while trimmed < 0:
        trimmed = trimmed + 2*math.pi
    while trimmed >= 2*math.pi:
        trimmed = trimmed - 2*math.pi
    return trimmed


def trimTheta_vec(theta):
    trimmed = theta
    # trimmed[trimmed < 0] += 2*math.pi
    # trimmed[trimmed >= 2*math.pi] -= 2*math.pi
    trimmed = np.fmod(2*math.pi + np.fmod(trimmed, 2*math.pi), 2*math.pi)
    return trimmed


if __name__ == "__main__":
    v = integrateGaussian(8.0, 3.0, 11.0, 14.0)
    print(v)
