# Created on Wed Aug 23 09:43:31 2017
# @author: tom
"""
performs clustering of data
call k_means(X, k, structure, method, version, fuzzyfier,
             iterations, C_in, U_in)
where
input: X numpy array nxd, matrix of n d-dimensional observations
       k positive integer, number of clusters
       structure list(int, list(floats), list(floats)),
                  number of non-hypertime dimensions, list of hypertime
                  radii nad list of wavelengths
       method string, defines type of initialization, possible ('random',
                                                                'prev_dim')
       version string, version of making weights (possible 'fuzzy',
                                                  'model', 'hard')
       fuzzyfier number, larger or equal one, not too large, usually 2 or 1
       iterations integer, max number of iterations
       C_in numpy array kxd, matrix of k d-dimensional cluster centres
                             from the last iteration
       U_in numpy array kxn, matrix of weights from the last iteration
and
output: C numpy array kxd, matrix of k d-dimensional cluster centres
        U numpy array kxn, matrix of weights
        densities numpy array kx1, matrix of number of
                measurements belonging to every cluster
especially
random initialization: does not need C_in and U_in and there can be any value
                       assigned
                       return randomly chosen points from dataset X as centres
prev_dim initialization: gets C_in as a known part of cluster centres and if it
                         realizes, that two new dimensions has been added to
                         the dataset, it randomly chooses points from
                         the circle in them - based on the information
                         in the variable structure
stable_init initialization: gets U_in as a weights from last clustering (above
                            smaller hypertime-space) and use them on X
                            transformed into new space to create new C (using
                            usual way)
fuzzy version: creates fuzzy partition matrix, recommended value for fuzzyfier
               is 2
hard version: creates crisp c-means partition matrix, recommended value for
              fuzzyfier is 1
model version: creates fuzzy partition matrix with values of weghts limited
               to eaual or less than 1
"""
import numpy as np
import dataset_io as dio


def k_means(X, k, structure, method, version, fuzzyfier,
            iterations, C_in, U_in):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           method string, defines type of initialization, possible ('random',
                                                                    'prev_dim')
           version string, version of making weights (possible 'fuzzy',
                                                      'model', 'hard')
           fuzzyfier number, larger or equal one, not too large, usually 2 or 1
           iterations integer, max number of iterations
           C_in numpy array kxd, matrix of k d-dimensional cluster centres
                                 from the last iteration
           U_in numpy array kxn, matrix of weights from the last iteration
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            densities numpy array kx1, matrix of number of
                    measurements belonging to every cluster
    uses: np.shape(), np.sum(),
          initialization(), distance_matrix(), partition_matrix(),
          new_centroids()
    objective: perform some kind of k-means
    """
    #print('starting clustering')
    d = np.shape(X)[1]
    J_old = 0
    C, U = initialization(X, k, method, C_in, U_in, structure, version)
    for iteration in range(iterations):
        D = distance_matrix(X, C, U, structure)
        U = partition_matrix(D, version)
        C = new_centroids(X, U, k, d, fuzzyfier)
        J_new = np.sum(U * D)
        if abs(J_old - J_new) < 0.01:
            #print('no changes! breaking loop.')
            break
        #if iteration % 10 == 0:
        #    print('iteration: ' + str(iteration))
        J_old = J_new
    densities = np.sum(U, axis=1, keepdims=True)
    #print('number of clustering iteration: ' + str(iteration))
#    print('output centres:')
#    print(list(C))
#    print('and densities:')
#    print(densities)
    #print('leaving clustering')
    return C, U, densities


def distance_matrix(X, C, U, structure):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           U numpy array kxn, matrix of weights
    output: D numpy array kxn, matrix of distances between every observation
            and every center
    uses: np.shape(), np.tile(), np.array(), np.abs()
    objective: to find difference between every observation and every
               center in every dimension
    """
    k, n = np.shape(U)
    D = []
    for cluster in range(k):
        C_cluster = np.tile(C[cluster, :], (n, 1))
        XC = dio.hypertime_substraction(X, C_cluster, structure)
        #XC = X - C_cluster
        # PROC ????
        ## L1 metrics
        #D.append(np.sum(np.abs(XC), axis=1))
        D.append(np.sqrt(np.sum(XC**2, axis=1)))
    D = np.array(D)
    return D


def partition_matrix(D, version):
    """
    input: D numpy array kxn, matrix of distances between every observation
           and every center
           version string, version of making weights (possible 'hard', 'fuzzy',
           'model')
    output: U numpy array kxn, matrix of weights
    uses: np.argmin(), np.sum(), np.zeros_like(), np.arange(),
          np.shape()
    objective: to create partition matrix (weights for new centroids
                                           calculation)
    """
    if version == 'fuzzy':
        U = 1 / (D + np.exp(-100))
        # muj pokus, jestli jsem to nemyslel jinak
        # U = U / np.sum(U, axis=0, keepdims=True)
    elif version == 'model':
        U = 1 / (D + np.exp(-100))
        U[D < 1] = 1
    elif version == 'hard':
        indices = np.argmin(D, axis=0)
        U = np.zeros_like(D)
        U[indices, np.arange(np.shape(D)[1])] = 1
    return U


def new_centroids(X, U, k, d, fuzzyfier):
    """
    input: U numpy array kxn, matrix of weights
           X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           d positive integer, number of dimensions
           fuzzyfier number, larger or equal one, not too large, usually 2 or 1
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
    uses: np.zeros(), np.tile(), np.sum()
    objective: calculate new centroids
    """
    U = U ** fuzzyfier
    C = np.zeros((k, d))
    for centroid in range(k):
        U_part = np.tile(U[centroid, :], (d, 1)).T
        C[centroid, :] = (np.sum(U_part * X, axis=0) / np.sum(U_part, axis=0))
    return C


def initialization(X, k, method, C_in, U_in, structure, version):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           method string, defines type of initialization, (possible 'random',
                                                           'prev_dim')
           C_in numpy array kxd, matrix of k d-dimensional cluster centres
                                 from the last iteration
           U_in numpy array kxn, matrix of weights from the last iteration
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           version string, version of making weights (possible 'fuzzy',
                                                      'model', 'hard')
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
    uses: np.shape(), np.random.choice(), np.arange(), np.random.randn(),
          np.shape(), np.empty(), np.c_[], np.cos(), np.sin(), np.zeros()
          distance_matrix(), partition_matrix()
    objective: create initial centroids and weights
    """
    if method == 'random':
        n, d = np.shape(X)
        if d < 1:
            print('unable to cluster, no data')
        else:
            #print('n: ' + str(n))
            #print('k: ' + str(k))
            C = X[np.random.choice(np.arange(n), size=k, replace=False), :]
            U = np.random.rand(k, n)
            D = distance_matrix(X, C, U, structure)
            U = partition_matrix(D, version)
    elif method == 'prev_dim':
        # supposing that the algorith adds only one circle per iteration
        d = np.shape(X)[1]
        C = np.empty((k, d))
        # known part of C
        d_in = np.shape(C_in)[1]
        C[:, : d_in] = C_in
        # unknown part of C lying randomly (R) on the circle with radius r
        if d_in + 2 == d:
            R = np.random.rand(k, 1)
            r = structure[1][-1]
            C[:, d_in:] = np.c_[r * np.cos(2*np.pi * R),
                                r * np.sin(2*np.pi * R)]
        U = np.empty_like(U_in)
        np.copyto(U, U_in)
    elif method == 'stable_init':
        C = new_centroids(X, U_in, k, d=np.shape(X)[1], fuzzyfier=1)
        U = np.empty_like(U_in)
        np.copyto(U, U_in)
    else:
        print('unknown method of initialization, returning zeros!')
        C = np.zeros((k, d))
    return C, U
