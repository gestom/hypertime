# Created on Sun Aug 27 13:11:12 2017
# @author: tom

"""
initializes the learning proces ("iteration 0").
call whole_initialization(path, k, edge_of_square, timestep, longest, shortest,
                          radius)
where
input: path string, path to file
       k positive integer, number of clusters
       edge_of_square float, spatial edge of cell in default units (meters)
       timestep float, time edge of cell in default units (seconds)
       longest float, legth of the longest wanted period in default
                      units
       shortest float, legth of the shortest wanted period
                       in default units
       radius float, size of radius of the first found hypertime circle
and
output: input_coordinates numpy array, coordinates for model creation
        overall_sum number (np.float64 or np.int64), sum of all measures
        structure list(int, list(floats), list(floats)),
                  number of non-hypertime dimensions, list of hypertime
                  radii nad list of wavelengths
        C numpy array kxd, matrix of k d-dimensional cluster centres
        U numpy array kxn, matrix of weights
        shape_of_grid numpy array dx1 int64, number of cells in every
                                             dimension
        time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                        over every
                                                        timeframe
        T numpy array shape_of_grid[0]x1, time positions of timeframes
        W numpy array Lx1, sequence of reasonable frequencies
        ES float64, squared sum of squares of residues from this iteration
        COV numpy array kxdxd, matrix of covariance matrices
        density_integrals numpy array kx1, matrix of ratios between
                                           measurements and grid cells
                                           belonging to the clusters
"""

import numpy as np
import dataset_io as dio
import grid
import fremen as fm
import model as mdl


def whole_initialization(training_data, k, edges_of_cell, longest,
                         shortest, training_dataset):
    """
    input: path string, path to file
           k positive integer, number of clusters
           edge_of_square float, spatial edge of cell in default units (meters)
           timestep float, time edge of cell in default units (seconds)
           longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
    output: input_coordinates numpy array, coordinates for model creation
            overall_sum number (np.float64 or np.int64), sum of all measures
            structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
            C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            shape_of_grid numpy array dx1 int64, number of cells in every
                                                 dimension
            time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
            T numpy array shape_of_grid[0]x1, time positions of timeframes
            W numpy array Lx1, sequence of reasonable frequencies
            ES float64, squared sum of squares of residues from this iteration
            COV numpy array kxdxd, matrix of covariance matrices
            density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
    uses: first_structure(), mdl.model_creation(), grid.time_space_positions(),
          first_time_frame_freqs(), fm.build_frequencies(), fm.chosen_period()
    objective: to perform first iteration step and to initialize variables
    """
    print('starting learning iteration: 0 (initialization)')
    structure = first_structure(training_data)
    input_coordinates, time_frame_sums, overall_sum, shape_of_grid, T,\
        valid_timesteps = grid.time_space_positions(edges_of_cell,
                                                    training_data,
                                                    training_dataset)
    if len(shape_of_grid[0]) == 1:
        hist_freqs = -1
        C = -1
        U = -1
        COV = -1
        density_integrals = -1
    else:
        hist_freqs, C, U, COV, density_integrals =\
            mdl.model_creation(input_coordinates, structure, training_data,
                               0, 0,  # C_in and U_in
                               k, shape_of_grid)
    time_frame_freqs = first_time_frame_freqs(overall_sum, shape_of_grid[0])
    W = fm.build_frequencies(longest, shortest)
    ES = -1  # no previous error
    P, W, ES, dES = fm.chosen_period(T, time_frame_sums,
                                     time_frame_freqs[0], W, ES,
                                     valid_timesteps)
    print('used structure: ' + str(structure))
    print('leaving learning iteration: 0 (initialization)')
    return input_coordinates, overall_sum, structure, C,\
        U, shape_of_grid, time_frame_sums, T, W, ES, P, COV,\
        density_integrals, valid_timesteps


def first_time_frame_freqs(overall_sum, shape_of_grid):
    """
    input: overall_sum number (np.float64 or np.int64), sum of all measures
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
    output: time_frame_freqs numpy array shape_of_grid[0]x1, sum of
                                                             frequencies
                                                             over every
                                                             timeframe
    uses: np.array()
    objective: to create first time_frame_freqs, i.e. time frames of a model
               that do not count the time
    """
    time_frame_freqs = np.array([overall_sum / shape_of_grid[0]] *
                                shape_of_grid[0])
    return time_frame_freqs


def first_structure(training_data):
    """
    input: path string, path to file
    output: structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    uses: np.shape(), dio.loading_data()
    objective: to create initial structure
    """
    dim = np.shape(training_data)[1] - 1
    structure = [dim, [], []]
    return structure
