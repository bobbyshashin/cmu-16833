import numpy as np
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


def init_particles_fixed_region(num_particles, occupancy_map, upper_left, lower_right):
    # init particles in a given retangular area (TODO)
    return None
