import numpy as np
import sys
import pdb

from MapReader import MapReader
from MapUtils import MapUtils
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling
from InitParticles import init_particles_freespace, init_particles_random, init_particles_fixed_region
from matplotlib import pyplot as plt
from matplotlib import figure as fig
import matplotlib.lines as mlines
import time
import math


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

    motion_model = MotionModel(1e-5, 1e-5, 1e-5, 1e-5)
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    # Enables visualization
    vis_flag = 1
    # Enables raycasting visualization for every frame
    vis_raycasting = 0
    # Re-precompute the raycasting lookup table
    precompute_raycasting = 0
    # Test raycasting and visualize with a single particle
    test_raycasting = 0
    # Test with a single particle at known start location
    test_one_particle = 0
    # Only test motion model
    test_motion_model_only = 0
    # Use vectorized motion model and measurement model
    use_vectorization = 1
    # Init mode: 0 = freespace, 1 = random, 2 = fixed region, -1 = do nothing
    init_mode = 0
    # init all particles with x=test_x, y=test_y, and random theta
    test_fix_xy = 0

    X_bar = None
    num_particles = 2000

    # initial starting location
    # test_x = 6500
    # test_y = 6000
    # test_theta = 0

    test_x = 4000
    test_y = 4150
    test_theta = 4.75

    if precompute_raycasting:
        sensor_model.precomputeRayCasting()
        return

    if test_one_particle:
        init_mode = -1
        use_vectorization = 0
        num_particles = 1
        X_bar = init_particles_freespace(num_particles, occupancy_map)
        # Test for one
        X_bar[:, 0] = test_x
        X_bar[:, 1] = test_y
        X_bar[:, 2] = test_theta

    if init_mode == 0:
        X_bar = init_particles_freespace(num_particles, occupancy_map)
    elif init_mode == 1:
        X_bar = init_particles_random(num_particles, occupancy_map)
    elif init_mode == 2:
        upper_left = np.array([0, 0])
        lower_right = np.array([300, 300])
        X_bar = init_particles_fixed_region(
            num_particles, occupancy_map, upper_left, lower_right)

    if test_raycasting:
        # Use the precompute table
        rays = sensor_model.rayCastingLookUp(
            np.array([test_x, test_y, test_theta]))
        # rays = sensor_model.rayCasting([test_x, test_y, test_theta])
        print(rays)
        map_utils.visualizeRays([test_x, test_y, test_theta], rays)
        plt.pause(20)
        return

    if vis_flag:
        map_utils.visualize_map()

    if test_fix_xy:
        X_bar[:, 0] = test_x
        X_bar[:, 1] = test_y
        # X_bar[0, 2] = test_theta

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

        if use_vectorization:
            """
                MOTION MODEL
            """
            X_t0 = X_bar[:, 0:3]
            X_t1 = motion_model.update_vec(u_t0, u_t1, X_t0)

            """
                SENSOR MODEL
            """
            if not test_motion_model_only:
                if (meas_type == "L"):
                    z_t = ranges
                    W_t, z_expected_arr = sensor_model.beam_range_finder_model_vec(
                        z_t, X_t1)
                    W_t = -1.0 / np.log10(W_t)
                    # print("z_t: ", z_t)
                    # print("z_expected_arr: ", z_expected_arr)
                    X_bar_new = np.hstack((X_t1, W_t.reshape(-1,1)))
                    # print("W_t: ", W_t/float(np.sum(W_t)))
                else: 
                    X_bar_new = np.hstack((X_t1, X_bar[:, 3].reshape(-1,1)))
            else:  # test with motion model only
                X_bar_new = np.hstack((X_t1, X_bar[:, 3].reshape(-1,1)))

        else:  # not vectorization
            for m in range(0, num_particles):

                """
                MOTION MODEL
                """
                x_t0 = X_bar[m, 0:3]
                x_t1 = motion_model.update(u_t0, u_t1, x_t0)

                if not test_motion_model_only:
                    """
                    SENSOR MODEL
                    """
                    if (meas_type == "L"):
                        z_t = ranges
                        # print("sensor model")
                        w_t, z_expected_arr = sensor_model.beam_range_finder_model(
                            z_t, x_t1)
                        print("z_t", z_t)
                        print("z_raycast", z_expected_arr)
                        if vis_raycasting:
                            map_utils.visualizeRays(
                                [x_t1[0], x_t1[1], x_t1[2]], z_expected_arr)
                            plt.pause(2)
                        # w_t = 1/num_particles
                        X_bar_new[m, :] = np.hstack((x_t1, w_t))
                    else:
                        X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
                else:
                    X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)
        # X_bar = resampler.multinomial_sampler(X_bar)

        if vis_flag:
            # if time_idx % 10 == 0:
            # print("vis")
            # visualize_map(occupancy_map)
            map_utils.visualize_timestep(X_bar, time_idx)


if __name__ == "__main__":
    main()
