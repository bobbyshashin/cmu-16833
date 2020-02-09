import numpy as np
from numpy import cos, sin
from scipy.integrate import quad
import scipy.stats

# integrates the gaussian distribution from x1 to x2
def integrateGaussian(mean, std, x1, x2):
    def normal_distribution_function(x):
        return scipy.stats.norm.pdf(x,mean,std)

    res, err = quad(normal_distribution_function, x1, x2)
    return res

def calcGaussian(mean, std, x):
    return scipy.stats.norm(mean, std).pdf(x)
    
def transformRobotToWorld(x, y, theta, x_robot_in_world, y_robot_in_world):
    xy = np.array([x, y]).reshape(2,1)
    trans = np.array([x_robot_in_world, y_robot_in_world]).reshape(2,1)
    R = np.array([[cos(theta), -sin(theta)], 
                  [sin(theta), cos(theta)]])
    return np.dot(R, xy) + trans


if __name__=="__main__":
    v = integrateGaussian(8.0, 3.0, 11.0, 14.0)
    print(v)


