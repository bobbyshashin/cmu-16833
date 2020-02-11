import numpy as np
import pdb

class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        weights = X_bar[:, -1]
        weights = weights/float(np.sum(weights))
        idxs = np.random.choice(num_particles, size=num_particles, p=weights)
        X_bar_resampled = X_bar[idxs, :]

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        weights = X_bar[:, -1]
        weights = weights/float(np.sum(weights))

        X_bar_resampled = []
        w_cumsum = np.cumsum(weights)
        r = np.random.rand() / num_particles
        c = weights[0]
        i = 0

        U = r + (np.arange(num_particles) / float(num_particles))

        for u in U:
            # print(u, c)
            while u > c:
                i += 1
                c += weights[i]
            X_bar_resampled.append(X_bar[i, :])
            # print(i)

        return np.array(X_bar_resampled)
        # return X_bar_resampled

if __name__ == "__main__":
    pass