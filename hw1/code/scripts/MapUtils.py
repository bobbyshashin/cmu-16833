import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import figure as fig
import Utils


class MapUtils:
    def __init__(self, occupancy_map, map_resolution):
        self._occupancy_map = occupancy_map
        self._resolution = map_resolution
        self._map_x = self._occupancy_map.shape[0]
        self._map_y = self._occupancy_map.shape[1]
        self._size_x = self._map_x * self._resolution
        self._size_y = self._map_y * self._resolution
        print('Finished reading 2D map of size: ' +
              '(' + str(self._size_x) + ',' + str(self._size_y) + ')')
        print('Map resolution: ' + str(self._resolution) + " cm")
        print('Occupancy map array shape: (' +
              str(self._map_x) + ',' + str(self._map_y) + ')')

    def visualize_map(self):
        fig = plt.figure()
        mng = plt.get_current_fig_manager()
        ax = plt.subplot(111)
        # The origin of matplotlib imshow is upper left corner
        # Axes are defined as:
        #  O --x-->
        #  |
        #  y
        #  |
        #  v
        #
        # Note: when imshow(matrix), matrix (row, col) -> plt (y, x)
        plt.ion()
        ax.imshow(self._occupancy_map, cmap='Greys', origin='upper')
        plt.axis([0, 800, 800, 0])

    def visualizeRays(self, robot_pose, laser_readings):
        plt.cla()
        ax = plt.subplot(111)
        plt.ion()
        ax.imshow(self._occupancy_map, cmap='Greys', origin='upper')
        plt.axis([0, 800, 800, 0])
        x = robot_pose[0] / 10
        y = robot_pose[1] / 10
        theta = robot_pose[2]-math.pi / 2.0
        # print("Robot pose: (" + str(x) + ", " + str(y) + ", " + str(theta))
        step_size = 5
        for i in range(0, 180, step_size):
            # may need to restrict to -pi to pi here
            curr_theta = theta + math.radians(i)
            curr_theta = Utils.trimTheta(curr_theta)
            new_x = x + laser_readings[i] * np.cos(curr_theta) / 10.0
            new_y = y + laser_readings[i] * np.sin(curr_theta) / 10.0
            new_x = int(min(self._map_x-1, max(new_x, 0)))
            new_y = int(min(self._map_y-1, max(new_y, 0)))
            # Note that the above (x, y) are essentially (row, col) in numpy array
            # So we convert it to (y, x) for plt to plot
            ax.plot([y, new_y], [x, new_x])
        plt.show()

    def visualize_timestep(self, X_bar, tstep):
        x_locs = X_bar[:, 0]/10.0
        y_locs = X_bar[:, 1]/10.0
        scat = plt.scatter(y_locs, x_locs, c='r', marker='o')
        plt.pause(0.00001)
        scat.remove()

    def get_map(self):
        return self._occupancy_map

    def get_map_size_x(self):  # in cm
        return self._size_x

    def get_map_size_y(self):  # in cm
        return self._size_y
