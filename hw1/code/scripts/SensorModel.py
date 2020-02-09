import numpy as np
import math
from math import cos, sin, degrees, radians
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
import pdb

from MapReader import MapReader
from Utils import integrateGaussian, calcGaussian


class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map, use_precomputed_rays=True):

        # take the range measurement data every 5 degrees
        self.resolution = 20
        self.range_stride = 10.0
        # from right to left, 180 degrees, counter-clockwise
        self.laser_fov = 180
        self.laser_max = 8191
        self.z_hit = 1.0
        self.z_short = 1.0
        self.z_max = 1.0
        self.z_rand = 1.0
        self.sigma_hit = 1.0
        self.lambda_short = 1.0

        # if 0 <= occupancy_map[i][j] < 0.3, we may regard cell (i, j) as freespace
        self.occupied_threshold = 0.1
        self.occupancy_map = occupancy_map
        self.map_resolution = 10.0
        self.map_x = occupancy_map.shape[0]
        self.map_y = occupancy_map.shape[1]

        self.use_precomputed_rays = use_precomputed_rays
        if use_precomputed_rays:
            self.raycasting_table = np.load(
                "../data/raycasting_table.npy")
            print("Loaded raycasting lookup table of size: ",
                  self.raycasting_table.shape)
            print(np.sum(self.raycasting_table))

    def precomputeRayCasting(self):
        all_readings = np.ones((self.map_x, self.map_y, 360),
                               dtype=np.float32) * self.laser_max
        for x in range(self.map_x):
            for y in range(self.map_y):
                for theta_deg in range(360):
                    theta_rad = radians(theta_deg)
                    max_range = int(self.laser_max / self.range_stride)+1
                    for i in range(max_range):
                        segment_length = self.range_stride * i
                        x_new = (x + 0.5) * self.map_resolution + \
                            segment_length * cos(theta_rad)
                        y_new = (y + 0.5) * self.map_resolution + \
                            segment_length * sin(theta_rad)

                        # ground_truth = self.rayCastingSingle(
                        # [x_new, y_new, theta_rad])
                        x_new = int(
                            min(self.map_x-1, max(x_new / self.map_resolution, 0)))
                        y_new = int(
                            min(self.map_y-1, max(y_new / self.map_resolution, 0)))

                        prob_occupied = self.occupancy_map[x_new, y_new]
                        if prob_occupied == -1 or prob_occupied >= self.occupied_threshold:
                            # if we hit an obstacle, return a reading (segment length)
                            # to handle -1? which is "unknown"
                            # unknown treated as obstacle for now
                            all_readings[x, y, theta_deg] = segment_length
                            # error = ground_truth - segment_length
                            # if np.abs(error) > 10:
                            #     print(error)
                            break
            print("Processed " + str(x) + " out of " + str(self.map_x) + " rows")

        np.save("../data/raycasting_table2.npy", all_readings)
        return

    def rayCastingSingle(self, laser_pose_in_map):
        x = laser_pose_in_map[0]
        y = laser_pose_in_map[1]
        theta = laser_pose_in_map[2]
        reading = self.laser_max

        max_range = int(self.laser_max / self.range_stride)+1
        for j in range(max_range):
            # compute line equation and extend the ray
            segment_length = self.range_stride * j

            x_new = x + segment_length * cos(theta)
            y_new = y + segment_length * sin(theta)

            x_new = int(
                min(self.map_x-1, max(x_new / self.map_resolution, 0)))
            y_new = int(
                min(self.map_y-1, max(y_new / self.map_resolution, 0)))
            # may vectorize here
            prob_occupied = self.occupancy_map[x_new, y_new]
            if prob_occupied == -1 or prob_occupied >= self.occupied_threshold:
                    # if we hit an obstacle, return a reading (segment length)
                    # to handle -1? which is "unknown"
                    # unknown treated as obstacle for now
                return segment_length
        return reading

    def rayCasting(self, laser_pose_in_map):
        if self.use_precomputed_rays:
            return self.rayCastingLookUp(laser_pose_in_map)
        print("Should not be here")
        x = laser_pose_in_map[0]
        y = laser_pose_in_map[1]

        readings = -np.ones(self.laser_fov)
        # iterate every angle from 0 to 180 degrees, right to left, counter-clockwise
        for i in range(self.laser_fov):
            theta_degree = degrees(laser_pose_in_map[2])
            # may need to truncate to -pi to pi here
            theta = radians(theta_degree + (self.laser_fov-i-1))

            max_range = int(self.laser_max / self.range_stride)+1
            for j in range(max_range):
                # compute line equation and extend the ray
                segment_length = self.range_stride * j

                x_new = x + segment_length * cos(theta)
                y_new = y + segment_length * sin(theta)

                x_new = int(
                    min(self.map_x-1, max(x_new / self.map_resolution, 0)))
                y_new = int(
                    min(self.map_y-1, max(y_new / self.map_resolution, 0)))
                # may vectorize here
                prob_occupied = self.occupancy_map[x_new, y_new]
                if prob_occupied == -1 or prob_occupied >= self.occupied_threshold:
                    # if we hit an obstacle, return a reading (segment length)
                    # to handle -1? which is "unknown"
                    # unknown treated as obstacle for now
                    readings[i] = segment_length
                    break
            if readings[i] == -1:
                readings[i] = self.laser_max
        return readings

    def rayCastingLookUp(self, laser_pose_in_map):
        x = int(
            min(self.map_x-1, max(laser_pose_in_map[0] / self.map_resolution, 0)))
        y = int(
            min(self.map_y-1, max(laser_pose_in_map[1] / self.map_resolution, 0)))

        theta_start = int(degrees(laser_pose_in_map[2]))
        theta_end = int(theta_start+self.laser_fov)
        split = (theta_end >= 360)
        if split:
            theta_end = theta_end - 360
        if not split:
            readings = self.raycasting_table[x, y, theta_start:theta_end]
        else:
            readings = np.append(
                self.raycasting_table[x, y, theta_start:-1], self.raycasting_table[x, y, 0:theta_end])
        readings = np.flip(readings)
        # print("Laser lookup shape: ", readings.shape)
        # print(x, y, theta_start)
        return readings

    def probHit(self, z_t, z_expected):
        if z_t >= 0 and z_t <= self.laser_max:
            normalizer = integrateGaussian(
                z_expected, self.sigma_hit, 0.0, self.laser_max)
            return normalizer * calcGaussian(z_expected, self.sigma_hit, z_t)
        else:
            return 0

    def probShort(self, z_t, z_expected):
        if z_t >= 0 and z_t <= z_expected:
            return self.lambda_short * np.exp(-self.lambda_short * z_t) / (1.0 - np.exp(-self.lambda_short * z_expected))
        else:
            return 0

    def probMax(self, z_t):
        if z_t == self.laser_max:
            # or maybe within some small threshold?
            return 1
        else:
            return 0

    def probRand(self, z_t):
        if z_t >= 0 and z_t <= self.laser_max:
            return 1.0 / self.laser_max
        else:
            return 0

    def computeBelief(self, z_t, z_expected):
        return self.z_hit * self.probHit(z_t, z_expected) + self.z_short * self.probShort(z_t, z_expected) + self.z_max * self.probMax(z_t) + self.z_rand * self.probRand(z_t)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        # (x, y, theta) of the robot (r) wrt the world frame (w)
        x_r_w = x_t1[0]
        y_r_w = x_t1[1]
        theta_r_w = x_t1[2]
        # (x, y, theta) of the laser (l) wrt the world frame (o)
        x_l_w = x_r_w + 25.0 * cos(theta_r_w)
        y_l_w = y_r_w + 25.0 * sin(theta_r_w)
        theta_l_w = theta_r_w

        pose_laser_map = np.array([x_l_w, y_l_w, theta_l_w])
        z_expected_arr = self.rayCasting(pose_laser_map)

        q = 1.0
        n = int(self.laser_fov / self.resolution)
        for i in range(n):
            k = 0 + self.resolution * i
            z = z_t1_arr[k]
            z_expected = z_expected_arr[i]
            q = q * self.computeBelief(z, z_expected)
        # print(q)
        return q, z_expected_arr


if __name__ == '__main__':
    pass
