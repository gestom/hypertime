# Created on Fri Jun  2 13:52:33 2017
# @author: tom

"""
There are three used functions, the first loads data from file, the second
transforms these data to the requested space-hypertime and the third function
returns directory of this file.
Other functions were built only for testing.

loading_data(path):
loads data from file on adress 'path' of structure (t, x, y, ...),
    the first dimension (variable, column) is understood as a time,
    others are measured variables in corresponding time.
    If there is only one column in the dataset, it is understood as positive
    occurences in measured times. Expected separator between values is SPACE
    (' ').

create_X(data, structure): create X from a loaded data as a data in hypertime,
                           where the structure of a space is derived from
                           the varibale structure
where
input: data numpy array nxd*, matrix of measures IRL, where d* is number
       of measured variables
       structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
and
output: X numpy array nxd, matrix of measures in hypertime
"""

import pandas as pd
import numpy as np
import pickle
import os


def loading_data(path):
    """
    input: path string, path to file
           path should be also a data, then it wil be send back
    output: data numpy array nxd*, matrix of measures IRL
    uses: pd.read_csv(), pd.values()
    objective: load data from file (t, x, y, ...), the first dimension
               (variable, column) is understood as a time, others are measured
               variables in corresponding time. If there is only one column
               in the dataset, it is understood as positive occurences in
               measured times. Expected separator between values is SPACE(' ').
    """
    if is_numpy_array(path):
        data = path # pointer
    else:
        postfix = path.rsplit('.', 1)[-1]
        if postfix == 'csv' or postfix == 'txt':
            df = pd.read_csv(path, sep=' ', header=None, index_col=None)
            data = df.values
        elif postfix == 'npy':
            data = np.load(path)
        else:
            print('unknown data type, returning zero')
            data = np.zeros(1)
    return data


def is_numpy_array(anything):
    """
    input: anything any data type
    output: answer boolean, True if anything is numpy array, False otherwise
    uses: 
    objective: to find out if anything is numpy array
    """
    return isinstance(anything, np.ndarray)


def create_X(data, structure):
    """
    input: data numpy array nxd*, matrix of measures IRL, where d* is number
                                  of measured variables
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    output: X numpy array nxd, matrix of measures in hypertime
    uses: np.empty(), np.c_[]
    objective: to create X as a data in hypertime, where the structure
               of a space is derived from the varibale structure
    """
    dim = structure[0]
    radii = structure[1]
    wavelengths = structure[2]
    #print(structure)
    #print(len(radii))
    ##### POKUS !!!
    X = np.empty((len(data), dim + len(radii) * 2))
    X[:, : dim] = data[:, 1: dim + 1]
    for period in range(len(radii)):
        r = radii[period]
        Lambda = wavelengths[period]
        X[:, dim: dim + 2] = np.c_[r * np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                   r * np.sin(data[:, 0] * 2 * np.pi / Lambda)]
        dim = dim + 2
    #X = np.empty((len(data), dim + 2))
    #X[:, : dim] = data[:, 1: dim + 1]
    #r = radii[-1]
    #Lambda = wavelengths[-1]
    #print('pouzita delka periody: ' + str(Lambda))
    #X[:, dim: dim + 2] = np.c_[r * np.cos(data[:, 0] * 2 * np.pi / Lambda),
    #                           r * np.sin(data[:, 0] * 2 * np.pi / Lambda)]
    ##### KONEC POKUSU !!!
    return X


def file_directory():
    """
    it is needed to call the file for '__file__' to be defined :)
    """
    return os.path.dirname(__file__)


def divide_dataset(dataset):
    """
    input: dataset numpy array, columns: time, vector of measurements, 0/1
                                  (occurence of event)
    output: training_data numpy array n*xd*, matrix of measures IRL, where
                                             d* is number of measured variables
                                             n* is 85% of measures
            evaluation_dataset numpy array, last 15% of dataset
            measured_time numpy array nX1, all measurement times
    uses: np.ceil(), np.split(),
    objective: to divide dataset to two parts (training and evaluation), 
               to create the "list" of all times of mesurements for fremen,
               to create training data of positive occurences of events
    """
    dividing_position = int(np.ceil(len(dataset) * 1))  # POKUS!!!
    training_dataset, evaluation_dataset =\
        np.split(dataset, [dividing_position])
    training_data = training_dataset[training_dataset[:, -1] == 1, 0: -1]
    # measured_times = training_dataset[:, 0]  # or [:, 0:1] ?
    ##### POKUS !!!
    #return training_data, evaluation_dataset, training_dataset
    return training_data, training_dataset, training_dataset
    ##### KONEC POKUSU !!!


def hypertime_substraction(X, Ci, structure):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           Ci_nxd numpy array nxd, matrix of n d-dimensional cluster centre
                                   copies
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    output: XC numpy array nxWTF, matrix of n WTF-dimensional substractions
    uses:
    objective: to substract C from X in hypertime
    """
    # classical difference
    # XC = X - Ci
    # cosine distance for hypertime and classical difference for space
    #non-hypertime dimensions substraction
    observations = np.shape(X)[0]
    ones = np.ones((observations, 1))
    dim = structure[0]
    radii = structure[1]
    XC = np.empty((observations, dim + len(radii)))
    XC[:, : dim] = X[:, : dim] - Ci[:, : dim]
    # hypertime dimensions substraction
    for period in range(len(radii)):
        r = radii[period]
        cos = (np.sum(X[:, dim + (period * 2): dim + (period * 2) + 2] *
                      Ci[:, dim + (period * 2): dim + (period * 2) + 2],
                      axis=1, keepdims=True) / (r ** 2))
        cos = np.minimum(np.maximum(cos, ones * -1), ones)
        XC[:, dim + period: dim + period + 1] = r * np.arccos(cos)
    return XC


# next there are functions used for testing only
def save_numpy_array(variable, name, save_directory):
    """
    input: variable numpy array, some variable
           name string, name of the file
           save_directory string, path to file, default 'variables'
    output: None
    uses: np.save()
    objective: to save numpy array
    """
    if '.' in name:
        parts = name.rsplit('.', 1)
        if parts[1] == '.npy':
            name = parts[0]
    np.save(save_directory + name + '.npy', variable)


def load_numpy_array(name, load_directory):
    """
    input: name string, name of the file
           load_directory string, path to file, default 'variables'
    output: variable numpy array, loaded variable
    uses: np.load()
    objective: to load numpy array
    """
    if '.' in name:
        parts = name.rsplit('.', 1)
        if parts[1] == '.npy':
            name = parts[0]
    return np.load(load_directory + name + '.npy')


def save_list(variable, name, save_directory):
    """
    input: variable numpy array, some variable
           name string, name of the file
           save_directory string, path to file, default 'variables'
    output: None
    uses: pickle.dump()
    objective: to save list
    """
    if '.' in name:
        parts = name.rsplit('.', 1)
        if parts[1] == '.lst':
            name = parts[0]
    with open(save_directory + name + '.lst', mode='wb') as myfile:
        pickle.dump(variable, myfile)


def load_list(name, load_directory):
    """
    input: name string, name of the file
           load_directory string, path to file, default 'variables'
    output: variable list (or int), loaded variable
    uses: pickle.load()
    objective: to load list
    """
    if '.' in name:
        parts = name.rsplit('.', 1)
        if parts[1] == '.lst':
            name = parts[0]
    with open(load_directory + name + '.lst', mode='rb') as myfile:
        return pickle.load(myfile)



def zobrazeni_do_rozumnych_souradnic(X, structure):
    Ci = create_zeros(structure)
    observations = np.shape(X)[0]
    ones = np.ones((observations, 1))
    dim = structure[0]
    radii = structure[1]
    XC = np.empty((observations, dim + len(radii)))
    XC[:, : dim] = X[:, : dim] - Ci[:, : dim]
    # hypertime dimensions substraction
    for period in range(len(radii)):
        r = radii[period]
        cos = (np.sum(X[:, dim + (period * 2): dim + (period * 2) + 2] *
                      Ci[:, dim + (period * 2): dim + (period * 2) + 2],
                      axis=1, keepdims=True) / (r ** 2))
        cos = np.minimum(np.maximum(cos, ones * -1), ones)
        XC[:, dim + period: dim + period + 1] = r * np.arccos(cos)
    return XC


def create_zeros(structure):
    """
    input: path string, path to file
           structure list(int, list(floats)), number of non-hypertime
                                              dimensions and list of hypertime
                                              radii
    output: X numpy array nxd, matrix of measures in hypertime
    uses: loading_data(), np.empty(), np.c_[]
    objective: to create X as a data in hypertime
    """
    # pouzivam puvodni funkci pro vytvoreni "pocatku souradnic" v pocatku dne
    dim = structure[0]
    radii = structure[1]
    wavelengths = structure[2]
    data = np.zeros((1, dim + len(radii)))
    X = np.empty((1, dim + len(radii) * 2))
    X[:, : dim] = data[:, 1: dim + 1]
    for period in range(len(radii)):
        r = radii[period]
        Lambda = wavelengths[period]
        X[:, dim: dim + 2] = np.c_[r * np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                   r * np.sin(data[:, 0] * 2 * np.pi / Lambda)]
        dim = dim + 2
    return X
