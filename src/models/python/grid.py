# Created on Mon Jul 17 15:14:57 2017
# @author: tom

"""
creates grid above data and outputs central positions of cels of grid
(input_coordinates), number of cells in every dimension (shape_of_grid),
time positions based on the grid (T), numbers of measurements in
timeframes (time_frame_sums) and number of measurements in all dataset
(overall_sum).
call time_space_positions(edge_of_square, timestep, path)
where
input: edge_of_square float, spatial edge of cell in default units (meters)
       timestep float, time edge of cell in default units (seconds)
       path string, path to file
and
output: input_coordinates numpy array, coordinates for model creation
        time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                        over every
                                                        timeframe
        overall_sum number (np.float64 or np.int64), sum of all measures
        shape_of_grid numpy array dx1 int64, number of cells in every
                                             dimension
        T numpy array shape_of_grid[0]x1, time positions of timeframes

timestep and edge_of_square has to be chosen based on desired granularity,
timestep refers to the time variable,
edge_of_square refers to other variables - it is supposed that the step
    (edge of cell) in every variable other than time is equal.
    If there are no other variables, some value has to be added but it is not
    used.
"""

import numpy as np
import dataset_io as dio


def time_space_positions(edges_of_cell, data, dataset):
    """
    input: edge_of_square float, spatial edge of cell in default units (meters)
           timestep float, time edge of cell in default units (seconds)
           path string, path to file
    output: input_coordinates numpy array, coordinates for model creation
            time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
            overall_sum number (np.float64 or np.int64), sum of all measures
            shape_of_grid numpy array dx1 int64, number of cells in every
                                                 dimension
            T numpy array shape_of_grid[0]x1, time positions of timeframes
    uses: loading_data(), number_of_edges(), hist_params(),
          cartesian_product()
    objective: to find central positions of cels of grid
    """
    extended_shape_of_grid = number_of_cells(dataset[:, 0:-1], edges_of_cell)
    if dio.is_numpy_array(data):
        central_points, time_frame_sums, overall_sum =\
            hist_params(data, extended_shape_of_grid)
    else:  # musim zmenit, je zbytecne to volat cele
        central_points, time_frame_sums, overall_sum =\
            hist_params(dataset[:, 0:-1], extended_shape_of_grid)
    input_coordinates = cartesian_product(*central_points)
    T = central_points[0]
    all_timesteps = np.histogramdd(dataset[:, 0:1],
                                   bins=extended_shape_of_grid[0][0:1],
                                   range=extended_shape_of_grid[1][0:1],
                                   normed=False, weights=None)[0]
    valid_timesteps = (all_timesteps > 0)
    T = T[valid_timesteps]
    return input_coordinates, time_frame_sums, overall_sum,\
        extended_shape_of_grid, T, valid_timesteps


def hist_params(data, extended_shape_of_grid):
    """
    input: X numpy array nxd, matrix of measures
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
    output: central_points list (floats), central points of cells
            time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
            overall_sum number (np.float64 or np.int64), sum of all measures
    uses: np.histogramdd(), np.arange(), np.shape(),np.sum()
    objective: find central points of cells of grid
    """
    histogram, edges = np.histogramdd(data, bins=extended_shape_of_grid[0],
                                      range=extended_shape_of_grid[1],
                                      normed=False, weights=None)
    central_points = []
    for i in range(len(edges)):
        step_lenght = (edges[i][-1] - edges[i][0]) / len(edges[i])
        central_points.append(edges[i][0: -1] + step_lenght / 2)
    osy = tuple(np.arange(len(np.shape(histogram)) - 1) + 1)
    time_frame_sums = np.sum(histogram, axis=osy)
    overall_sum = np.sum(time_frame_sums)
    return central_points, time_frame_sums, overall_sum


def number_of_cells(data, edges_of_cell):
    """
    input: X numpy array nxd, matrix of measures
           edge_of_square float, length of the edge of 2D part of a "cell"
           timestep float, length of the time edge of a "cell"
    output: shape_of_grid numpy array, number of edges on t, x, y, ... axis
    uses:np.shape(), np.max(), np.min(),np.ceil(), np.int64()
    objective: find out number of cells in every dimension
    """
    # number of predefined cubes in the measured space
    # changed to exact length of timestep and edge of square
    # changed to general shape of cell
    extended_shape_of_grid = [[],[]]
    n, d = np.shape(data)
    for i in range(d):
        min_i = np.min(data[:, i])
        max_i = np.max(data[:, i])
        range_i = max_i - min_i
        edge_i = float(edges_of_cell[i])
        number_of_bins = np.floor(range_i / edge_i) + 1
        half_residue = (1 - (range_i % edge_i)) / 2.0
        position_min =  min_i - half_residue
        position_max =  max_i + half_residue
        extended_shape_of_grid[0].append(int(number_of_bins))
        extended_shape_of_grid[1].append([position_min, position_max])
    return extended_shape_of_grid


def cartesian_product(*arrays):
    """
    downloaded from:
    'https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of'+\
    '-x-and-y-array-points-into-single-array-of-2d-points'
    input: *arrays enumeration of central_points
    output: numpy array (central positions of cels of grid)
    uses: np.empty(),np.ix_(), np.reshape()
    objective: to perform cartesian product of values in columns
    """
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la],
                   dtype=arrays[0].dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
