# Created on Tue Jul 25 15:43:06 2017
# @author: tom

"""
returns model parameters and histogram above time-space
call model_creation(input_coordinates, structure, path, C_old,
                    U_old, k, shape_of_grid)
where
input: input_coordinates numpy array, coordinates for model creation
       structure list(int, list(floats), list(floats)),
                  number of non-hypertime dimensions, list of hypertime
                  radii nad list of wavelengths
       path string, path to file
       C_old numpy array kxd, centres from last iteration
       U_old numpy array kxn, weights from last iteration
       k positive integer, number of clusters
       shape_of_grid numpy array dx1 int64, number of cells in every
                                            dimension
and
output: hist_freqs numpy array (shape_of_grid), multidimensional histogram
                                                of frequencies(stat) of
                                                a model over the grid
        C numpy array kxd, matrix of k d-dimensional cluster centres
        U numpy array kxn, matrix of weights
        COV numpy array kxdxd, matrix of covariance matrices
        density_integrals numpy array kx1, matrix of ratios between
                                           measurements and grid cells
                                           belonging to the clusters
"""
# COV neni COVARIANCE MATRIX ale PRECISION MATRIX !!!!!
import dataset_io as dio
import clustering as cl
import numpy as np
import gc


def model_creation(input_coordinates, structure, data, C_old, U_old, k,
                   shape_of_grid):
    """
    input: input_coordinates numpy array, coordinates for model creation
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           path string, path to file
           C_old numpy array kxd, centres from last iteration
           U_old numpy array kxn, weights from last iteration
           k positive integer, number of clusters
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
    output: hist_freqs numpy array (shape_of_grid), multidimensional histogram
                                                    of frequencies(stat) of
                                                    a model over the grid
            C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            COV numpy array kxdxd, matrix of covariance matrices
            density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
    uses: model_parameters(), coordinates_densities(), frequencies()
          np.reshape()
    objective: to create grid of frequencies(stat) over time-space (histogram),
               pass centres and weights to the next clusters initialization,
               and return model parameters (C, COV, density_integrals)
    """
    C, U, COV, densities = model_parameters(data, structure, C_old, U_old, k)
    grid_densities = coordinates_densities(input_coordinates, C, COV,
                                           structure, k)
    density_integrals = densities / grid_densities
    freqs = frequencies(input_coordinates, C, COV,
                        structure, k, density_integrals)
    hist_freqs = freqs.reshape(shape_of_grid[0])
    return hist_freqs, C, U, COV, density_integrals


def model_parameters(data, structure, C_old, U_old, k):
    """
    input: path string, path to file
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           C_old numpy array kxd, centres from last iteration
           U_old numpy array nxd, weights from last iteration
           k positive integer, number of clusters
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            COV numpy array kxdxd, matrix of covariance matrices
            densities numpy array kx1, matrix of number of measurements
                                       belonging to every cluster
    uses: dio.create_X(), cl.k_means(), covariance_matrices()
    objective: to find model parameters
    """
    X = dio.create_X(data, structure)
    # test to find out if clusters are known from previous clustering
    try:
        len(U_old)
        ##### POKUS !!!
        #used_method = 'stable_init'  # originaly 'prev_dim'
        used_method = 'random'
        ##### KONEC POKUSU !!!
    except TypeError:
        used_method = 'random'
    #print('type of initialization for clustering: ' + used_method)
    C, U, densities = cl.k_means(X, k, structure,
                                 method=used_method,
                                 version='hard',  # weight calculation
                                 fuzzyfier=1,  # weighting exponent
                                 iterations=100,
                                 C_in=C_old, U_in=U_old)
    COV = covariance_matrices(X, C, U, structure)
    return C, U, COV, densities


def covariance_matrices(X, C, U, structure):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           U numpy array kxn, matrix of weights
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    output: COV numpy array kxdxd, matrix of covariance matrices
    uses: cl.distance_matrix(), cl.partition_matrix()
          np.shape(), np.tile(), np.cov(), np.linalg.inv(), np.array()
    objective: to calculate covariance matrices for model
    """
    k, n = np.shape(U)
    D = cl.distance_matrix(X, C, U, structure)
    ## not pure fuzzy W :)
    #W = cl.partition_matrix(D, version='fuzzy')
    ## W with binary memberships from U
    #W = W * U
    COV = []
    for cluster in range(k):
        C_cluster = np.tile(C[cluster, :], (n, 1))
        XC = dio.hypertime_substraction(X, C_cluster, structure)
        #XC = X - C_cluster
        #V = np.cov(XC, aweights=W[cluster, :], ddof=0, rowvar=False)
        #V = np.cov(XC, ddof=0, rowvar=False)  # puvodni, melo by byt stejne
        V = np.cov(XC, bias=True, rowvar=False)
        if len(np.shape(V)) == 2:
            Vinv = np.linalg.inv(V)
        else:
            #print('V: ' + str(V))
            Vinv = np.array([[1 / V]])
        COV.append(Vinv)
    COV = np.array(COV)
    return COV


def coordinates_densities(input_coordinates, C, COV, structure, k):
    """
    input: input_coordinates numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
    output: grid_densities numpy array kx1, number of cells belonging to the
                                            clusters
    uses: iter_over_coordinates()
          np.shape(), np.zeros(), np.empty(), gc.collect()
    objective: to call iter_over_coordinates() above smaller parts
               of input_coordinates (every part of (5e7 / k) lines)
               and to find out the number of cells belonging to the clusters
    """
    number_of_coordinates = np.shape(input_coordinates)[0]
    volume_of_data = number_of_coordinates * k
    number_of_parts = (volume_of_data // int(5e7)) + 1
    length_of_part = number_of_coordinates // (number_of_parts)
    finish = 0
    grid_densities = np.zeros((k, 1))
    for i in range(number_of_parts):
        start = i * length_of_part
        finish = (i + 1) * length_of_part - 1
        grid_densities_part =\
            iter_over_coordinates(input_coordinates[start: finish, :], C, COV,
                                  structure, k)
        grid_densities += grid_densities_part
        gc.collect()
    grid_densities_part = iter_over_coordinates(input_coordinates[finish:, :],
                                                C, COV, structure, k)
    grid_densities += grid_densities_part
    return grid_densities


def iter_over_coordinates(input_coordinates_part, C, COV, structure, k):
    """
    input: input_coordinates_part numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
    output: grid_densities_part numpy array kx1, number of part of cells
                                                 belonging to the clusters
    uses: dio.create_X(), np.shape(), gc.collect(), np.tile(),
          np.sum(), np.dot(), np.array(),
          cl.partition_matrix()
    objective: to find out the number of cells (part of them) belonging to
               the clusters
    """
    X = dio.create_X(input_coordinates_part, structure)
    n, d = np.shape(X)
    gc.collect()
    D = []
    for cluster in range(k):
        C_cluster = np.tile(C[cluster, :], (n, 1))
        XC = dio.hypertime_substraction(X, C_cluster, structure)
        #XC = X - C_cluster
        VI = COV[cluster]#COV[cluster, :, :]
        D.append(np.sum(np.dot(XC, VI) * XC, axis=1))
        gc.collect()
    D = np.array(D)
    gc.collect()
    U = cl.partition_matrix(D, version='model')
    U = U ** 2
    gc.collect()
    grid_densities_part = np.sum(U, axis=1, keepdims=True)
    return grid_densities_part


def frequencies(input_coordinates, C, COV, structure, k,
                density_integrals):
    """
    input: input_coordinates numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
           density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
    output: freqs numpy array len(input_coordinates_part)x1,
                                           frequencies(stat) obtained
                                           from model in positions
                                           of input_coordinates
    uses: iter_over_freqs()
          np.shape(), np.zeros(), np.empty(), gc.collect(), np.reshape()
    objective: to call iter_over_freqs() above smaller parts
               of input_coordinates (every part of (5e7 / k) lines)
               and to create grid of frequencies(stat) over time-space
               (histogram)
    """
    number_of_coordinates = np.shape(input_coordinates)[0]
    volume_of_data = number_of_coordinates * k
    number_of_parts = (volume_of_data // int(5e7)) + 1
    length_of_part = number_of_coordinates // (number_of_parts)
    finish = 0
    freqs = np.empty(number_of_coordinates)
    for i in range(number_of_parts):
        start = i * length_of_part
        finish = (i + 1) * length_of_part - 1
        freqs_part = iter_over_freqs(input_coordinates[start: finish, :],
                                     C, COV, structure, k,
                                     density_integrals)
        freqs[start: finish] = freqs_part
        gc.collect()
    freqs_part = iter_over_freqs(input_coordinates[finish:, :],
                                 C, COV, structure, k,
                                 density_integrals)
    freqs[finish:] = freqs_part
#    hist_freqs = freqs.reshape(shape_of_grid)
#    return hist_freqs
    return freqs


def iter_over_freqs(input_coordinates_part, C, COV, structure, k,
                    density_integrals):
    """
    input: input_coordinates_part numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
           density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
    output: freqs_part numpy array len(input_coordinates_part)x1,
                                           frequencies(stat) obtained
                                           from model in positions of part
                                           of input_coordinates
    uses: dio.create_X(), np.shape(), gc.collect(), np.tile(),
          np.sum(), np.dot(), np.array(),
          cl.partition_matrix()
    objective: to create grid of frequencies(stat) over a part time-space
               (histogram)
    """
    X = dio.create_X(input_coordinates_part, structure)
    n, d = np.shape(X)
    D = []
    for cluster in range(k):
        C_cluster = np.tile(C[cluster, :], (n, 1))
        XC = dio.hypertime_substraction(X, C_cluster, structure)
        #XC = X - C_cluster
        VI = COV[cluster]#COV[cluster, :, :]
        D.append(np.sum(np.dot(XC, VI) * XC, axis=1))
        gc.collect()
    D = np.array(D)
    gc.collect()
    U = cl.partition_matrix(D, version='model')
    U = (U ** 2) * density_integrals
    gc.collect()
    freqs_part = np.sum(U, axis=0)
    return freqs_part


def one_freq(one_input_coordinate, C, COV, structure, k,
                    density_integrals):
    """
    input: one_input_coordinate numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
           density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
    output: freq array len(input_coordinates_part)x1,
                                           frequencies(stat) obtained
                                           from model in positions of part
                                           of input_coordinates
    uses: dio.create_X(), np.shape(), gc.collect(), np.tile(),
          np.sum(), np.dot(), np.array(),
          cl.partition_matrix()
    objective: to create grid of frequencies(stat) over a part time-space
               (histogram)
    """
    X = dio.create_X(one_input_coordinate, structure)
    n, d = np.shape(X)
    D = []
    for cluster in range(k):
        # C_cluster = np.tile(C[cluster, :], (n, 1))
        XC = dio.hypertime_substraction(X, C[cluster:cluster+1], structure)
        #XC = X - C[cluster]
        VI = COV[cluster]#COV[cluster, :, :]
        D.append(np.sum(np.dot(XC, VI) * XC, axis=1))
        # gc.collect()
    D = np.array(D)
    # gc.collect()
    U = cl.partition_matrix(D, version='model')
    U = (U ** 2) * density_integrals
    # gc.collect()
    freq = np.sum(U, axis=0)
    return freq
