import numpy as np
import sys
import pdb

from MapReader import MapReader
from MapUtils import MapUtils
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import matplotlib.lines as mlines
import time
import math


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    map_resolution = 10

    # initialize [x, y, theta] positions in world_frame for all particles
    freespace = np.argwhere(occupancy_map == 0)
    freespace = freespace * map_resolution
    # freespace[:,[0,1]] = freespace[:,[1,0]]
    rand_idx = np.random.uniform(0, len(freespace), num_particles).astype(int)
    xy0_vals = freespace[rand_idx]
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((xy0_vals, theta0_vals, w0_vals))

    return X_bar_init


def main():
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """

    """
    Initialize Parameters
    """
    src_path_map = '../data/map/wean.dat'
    src_path_log = '../data/log/robotdata1.log'

    map_reader = MapReader(src_path_map)
    occupancy_map, map_resolution = map_reader.readMap()
    map_utils = MapUtils(occupancy_map, map_resolution)
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel(0.5, 0.5, 0.5, 0.5)
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = 100
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    vis_flag = 1
    precompute_raycasting = 0
    test_raycasting = 0

    if test_raycasting:
        test_x = 4000
        test_y = 4000
        test_theta = 2.0
        rays = sensor_model.rayCastingLookUp(
            np.array([test_x, test_y, test_theta]))
        # rays = sensor_model.rayCasting([test_x, test_y, test_theta])
        # print(rays)
        map_utils.visualizeRays([test_x, test_y, test_theta], rays)
        plt.pause(20)
        return
    if vis_flag:
        map_utils.visualize_map()

    if precompute_raycasting:
        sensor_model.precomputeRayCasting()
        return
    """
    Monte Carlo Localization Algorithm : Main Loop
    """

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]
        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging)
        # continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]
        # else:
        #     continue

        print("Processing time step " + str(time_idx) +
              " at time " + str(time_stamp) + "s")

        if (first_time_idx):
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        """
            MOTION MODEL
        """
        X_t0 = X_bar[:, 0:3]
        X_t1 = motion_model.update_vec(u_t0, u_t1, X_t0)

        for m in range(0, num_particles):

            """
            MOTION MODEL
            """
            # x_t0 = X_bar[m, 0:3]
            # x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            # map_utils.visualizeRays(
            #     [x_t1[0], x_t1[1], x_t1[2]], z_expected_arr)
            # plt.pause(2)
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                # print("sensor model")
                w_t, z_expected_arr = sensor_model.beam_range_finder_model(
                    z_t, X_t1[m,:])

                # map_utils.visualizeRays(
                #     [x_t1[0], x_t1[1], x_t1[2]], z_expected_arr)
                # plt.pause(2)
                # w_t = 1/num_particles
                X_bar_new[m, :] = np.hstack((X_t1[m,:], w_t))
            else:
                X_bar_new[m, :] = np.hstack((X_t1[m,:], X_bar[m, 3]))
            # X_bar_new[m,:] = np.hstack((x_t1, X_bar[m,3]))
        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        # X_bar = resampler.low_variance_sampler(X_bar)
        X_bar = resampler.multinomial_sampler(X_bar)

        if vis_flag:
            # if time_idx % 10 == 0:
            print("vis")
            # visualize_map(occupancy_map)
            map_utils.visualize_timestep(X_bar, time_idx)


if __name__ == "__main__":
    main()
