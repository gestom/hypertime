import numpy as np
import model as mdl
import grid


def evaluation_step(evaluation_dataset, C, COV, density_integrals,\
                    structure, k, edges_of_cell):
    """
    """
    evaluation_data = evaluation_dataset[evaluation_dataset[:, -1] == 1, 0: -1]
    freqs, input_coordinates, extended_shape_of_grid, valid_timesteps =\
        params_for_model(evaluation_dataset, C, COV, density_integrals,\
                         structure, k, edges_of_cell, evaluation_data)
    model = np.histogramdd(input_coordinates,\
                           bins=extended_shape_of_grid[0],
                           range=extended_shape_of_grid[1],\
                           normed=False, weights=freqs)[0]
    reality = np.histogramdd(evaluation_data,\
                             bins=extended_shape_of_grid[0],
                             range=extended_shape_of_grid[1],\
                             normed=False, weights=None)[0]
    diff = (np.sum((reality - model)[valid_timesteps] ** 2)) ** 0.5
    """
    model_filename = '../out/model' + str(len(structure[1])) + '.txt'
    reality_filename = '../out/reality' + str(len(structure[1])) + '.txt'
    with open(model_filename, 'w') as mf:
        np.savetxt(mf, model)
    with open(reality_filename, 'w') as rf:
        np.savetxt(rf, reality)
    """
    #print('the difference between model and evaluation data is:')
    #print(diff)
    return diff



def params_for_model(evaluation_dataset, C, COV, density_integrals,
                     structure, k, edges_of_cell, evaluation_data):
    """
    input: path string, path to file
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           density_integrals numpy array kx1, matrix of ratios between
                                              measurements and grid cells
                                              belonging to the clusters
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
           edge_of_square float, spatial edge of cell in default units (meters) 
           timestep float, time edge of cell in default units (seconds)
    output: freqs numpy array len(input_coordinates_part)x1,
                                           frequencies(stat) obtained
                                           from model in positions
                                           of input_coordinates
            input_coordinates numpy array, coordinates for model creation
            shape_of_grid numpy array dx1 int64, number of cells in every
                                                 dimension
    uses: grid.time_space_positions(), mdl.frequencies()
    objective: to return frequencies (stat) on coordinates in the chosen grid
    """
    evaluation_data = evaluation_dataset[evaluation_dataset[:, -1] == 1, 0: -1]
    input_coordinates, time_frame_sums, overall_sum,\
        extended_shape_of_grid, T, valid_timesteps =\
        grid.time_space_positions(edges_of_cell, evaluation_data,
                                  evaluation_dataset)
    freqs = mdl.frequencies(input_coordinates, C, COV,
                            structure, k, density_integrals)
    return freqs, input_coordinates, extended_shape_of_grid, valid_timesteps
