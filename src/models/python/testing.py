# Created on Wed Aug 30 16:46:21 2017
# @author: tom


"""
returns model (a grid of frequencies(stat)) above chosen space.
if the chosen space is based on data, returns also grid of measured values and
    prints differences between measured values model, zeros and averages.

call model_above_space(C, COV, density_integrals, structure, k,
                       edge_of_square, timestep,
                       space_limits):
where
input: C numpy array kxd, matrix of k d-dimensional cluster centres
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
       space_limits list(float or int), boarders of chosen space for model
and
output: model numpy array (shape_of_grid), grid of modeled frequencies(stat)

OR

call model_above_data(path, C, COV, density_integrals, structure, k,
                      edge_of_square, timestep, prefix, visualise, average):
where
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
       prefix string, string to differ model visualisations
       visualise boolean, to create graph or not
       average float, average value per cell in testing data
and
output: model numpy array (shape_of_grid), grid of modeled frequencies(stat)
        reality numpy array (shape_of_grid), grid of measured frequencies(stat)
"""
import numpy as np
import os

import dataset_io as dio
import grid
import model as mdl
#import get_directory as gd

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.misc import toimage


def model_above_data(path, C, COV, density_integrals, structure, k,
                     edge_of_square, timestep, prefix, visualise, average):
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
           prefix string, string to differ model visualisations
           visualise boolean, to create graph or not
           average float, average value per cell in testing data
    output: model numpy array (shape_of_grid), grid of modeled
                                               frequencies(stat)
            reality numpy array (shape_of_grid), grid of measured
                                                 frequencies(stat)
    uses: params_for_model(), dio.loading_data(),
          difference_visualisation_3d(), difference_visualisation_2d(),
          np.ones_like(), np.zeros_like(), np.histogramdd(), np.sum(),
          np.max(), np.min()
    objective:
    """
    freqs, input_coordinates, shape_of_grid =\
        params_for_model(path, C, COV, density_integrals, structure, k,
                         edge_of_square, timestep)
    data = dio.loading_data(path)
    average = np.ones_like(freqs) * average
    zero = np.zeros_like(freqs)
    model = np.histogramdd(input_coordinates, bins=shape_of_grid,
                           range=None, normed=False, weights=freqs)[0]
    reality = np.histogramdd(data, bins=shape_of_grid,
                             range=None, normed=False, weights=None)[0]
    zeros = np.histogramdd(input_coordinates, bins=shape_of_grid,
                           range=None, normed=False, weights=zero)[0]
    averages = np.histogramdd(input_coordinates, bins=shape_of_grid,
                              range=None, normed=False, weights=average)[0]
    diff = (np.sum((reality - model) ** 2)) ** 0.5
    error_of_averages = (np.sum((reality - averages) ** 2)) ** 0.5
    error_of_zeros = (np.sum((reality - zeros) ** 2)) ** 0.5
    if visualise:
        if len(shape_of_grid) == 3:
            hours_of_measurement = (np.max(data[:, 0]) - np.min(data[:, 0]))\
                                   / (60*60)
            starting_hour = (np.min(data[:, 0]) % (60*60*24)) / (60*60*24)
            difference_visualisation_3d(model, reality, shape_of_grid,
                                        hours_of_measurement, starting_hour,
                                        prefix)
        elif len(shape_of_grid) == 2:
            difference_visualisation_2d(model, reality, shape_of_grid, prefix)
    print('\nshape of grid [t, x, y]: ' + str(shape_of_grid))
    print('distance between measurement and model: ' + str(diff))
    print('distance between measurement and zeros: ' + str(error_of_zeros))
    print('distance between measurement and mean value: ' + (error_of_averages))
    return model, reality


def model_above_space(C, COV, density_integrals, structure, k,
                      edge_of_square, timestep,
                      space_limits):
    """
    input: C numpy array kxd, matrix of k d-dimensional cluster centres
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
           space_limits list(float or int), boarders of chosen space for model
    output: model numpy array (shape_of_grid), grid of modeled
                                               frequencies(stat)
    uses: gd.file_directory(), params_for_model()
          np.array(), os.path.join(), np.save(), np.histogramdd()
    objective: to create model above chosen space
    """
    data = np.array(space_limits)
    directory = dio.file_directory()
    path = os.path.join(directory, '..', 'tmp', 'limits.npy')
    np.save(path, data)
    freqs, input_coordinates, shape_of_grid =\
        params_for_model(path, C, COV, density_integrals, structure, k,
                         edge_of_square, timestep)
    model = np.histogramdd(input_coordinates, bins=shape_of_grid,
                           range=None, normed=False, weights=freqs)[0]
    return model


def params_for_model(path, C, COV, density_integrals,
                     structure, k, edge_of_square, timestep):
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
    input_coordinates, time_frame_sums, overall_sum, shape_of_grid, T =\
        grid.time_space_positions(edge_of_square, timestep, path)
    freqs = mdl.frequencies(input_coordinates, C, COV,
                            structure, k, density_integrals)
    return freqs, input_coordinates, shape_of_grid


def difference_visualisation_3d(H_probs, H_train, shape_of_grid,
                                hours_of_measurement, starting_hour, prefix):
    """
    input:
    output:
    uses:
    objective:
    """
    directory = dio.file_directory()
    fig = plt.figure(dpi=400)
    for i in range(shape_of_grid[0]):
        # measured data
        plt.subplot(121)
        cmap = mpl.colors.ListedColormap(['black', 'red', 'orange', 'pink',
                                          'yellow', 'white', 'lightblue'])
        bounds = [-0.5, 0.5, 1.5, 3.5, 10.5, 20.5, 50.5, 100.5]
        my_norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(H_train[i, :, :], interpolation='nearest',
                         cmap=cmap, norm=my_norm)
        plt.colorbar(img, cmap=cmap,
                     norm=my_norm, boundaries=bounds,
                     ticks=[0.5, 1.5, 3.5, 10.5, 20.5, 50.5, 100.5],
                     fraction=0.046, pad=0.01)
        plt.xticks(list(np.arange(4)*shape_of_grid[2]/4))
        plt.yticks(list(np.arange(4)*shape_of_grid[1]/4))
        # model
        plt.subplot(122)
        bounds = [-0.5, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128,
                  0.265, 0.512, 1.024]
        my_norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        img = plt.imshow(H_probs[i, :, :], interpolation='nearest',
                         cmap='Greys', norm=my_norm)
        plt.colorbar(img, cmap='Greys',
                     norm=my_norm, boundaries=bounds,
                     fraction=0.046, pad=0.01)
        plt.xticks(list(np.arange(4)*shape_of_grid[1]/4))
        plt.yticks(list(np.arange(4)*shape_of_grid[2]/4))
        # together
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        # name the file, assuming thousands of files
        times = i / (shape_of_grid[0] / hours_of_measurement) + starting_hour
        hours = times % 24
        days = int(times / 24)
        hours = str(hours)
        days = str(days)
        if len(hours.split('.')[0]) == 1:
            hours = '0' + hours
        if len(hours.split('.')[1]) == 1:
            hours = hours + '0'
        if len(hours.split('.')[1]) > 2:
            hours = hours.split('.')[0] + '.' + hours.split('.')[1][:2]
        if len(days.split('.')[0]) == 1:
            days = '0' + days
        name = str(i) + '.' + days + '.' + hours
        if len(name.split('.')[0]) == 1:
            name = '0' + name
        if len(name.split('.')[0]) == 2:
            name = '0' + name
        if len(name.split('.')[0]) == 3:
            name = '0' + name
        name = prefix + name
        path = os.path.join(directory, '..', 'fig', name + '.png')
        fig.canvas.draw()
        # really do not understand :) coppied from somewhere
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.clf()
        # save to file
        toimage(data).save(path)
    plt.close(fig)


def difference_visualisation_2d(model, reality, shape_of_grid, prefix):
    """
    input:
    output:
    uses:
    objective:
    """
    # build picture and save it
    fig = plt.figure(dpi=400)
    # training data
    plt.subplot(121)
    cmap = mpl.colors.ListedColormap(['black', 'red', 'orange', 'yellow'])
    bounds = [-0.5, 0.64, 1.28, 2.56, 5.12]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    img = plt.imshow(reality[:, :], interpolation='nearest',
                     cmap=cmap, norm=norm)
    plt.xticks([])
    plt.yticks([])

    plt.xticks([0, 6, 12, 18])
    plt.yticks([0, 24, 48, 72, 96, 120, 144, 168])
    # model
    plt.subplot(122)
    cmap = mpl.colors.ListedColormap(['black', 'white', 'lightblue', 'blue',
                                      'purple', 'red', 'orange', 'yellow'])
    bounds = [-0.5, 0.02, 0.04, 0.08, 0.32, 0.64, 1.28, 2.56, 5.12]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    img = plt.imshow(model[:, :], interpolation='nearest',
                     cmap=cmap, norm=norm)
    plt.colorbar(img, cmap=cmap,
                 norm=norm, boundaries=bounds,
                 ticks=[0.02, 0.04, 0.08, 0.32, 0.64, 1.28, 2.56, 5.12],
                 fraction=0.046, pad=0.01)
    plt.xticks([0, 6, 12, 18])
    plt.yticks([0, 24, 48, 72, 96, 120, 144, 168])
    # all together
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    # name the file
    name = prefix
    directory = dio.file_directory()
    path = os.path.join(directory, '..', 'fig', name + '.png')
    fig.canvas.draw()
    # really do not understand :) coppied from somewhere
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    # save
    toimage(data).save(path)
    plt.close(fig)












