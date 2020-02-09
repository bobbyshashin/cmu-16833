import numpy as np

class MapReader:
    def __init__(self, src_path_map):
        self._map_path = src_path_map

    def readMap(self):
        # Read from map file path
        occupancy_map = np.genfromtxt(self._map_path, skip_header=7)

        # -1 as unknown spaces
        occupancy_map[occupancy_map < 0] = -1
        # Original file stores "probability of freespace"
        # Here we convert to "probability of occupied"
        occupancy_map[occupancy_map > 0] = 1 - occupancy_map[occupancy_map > 0]

        # Each cell has a 10cm resolution in x,y axes
        resolution = 10

        return occupancy_map, resolution