#!/usr/bin/env python2
# Created on Wed Aug 30 16:54:05 2017
# @author: tom

"""
list of manual parameters:
longest float, legth of the longest wanted period in default
               units
shortest float, legth of the shortest wanted period
                in default units
path_train string, path to training file
edge_of_square float, spatial edge of cell in default units (meters)
timestep float, time edge of cell in default units (seconds)
k positive integer, number of clusters
radius float, size of radius of the first found hypertime circle
number_of_periods int, max number of added hypertime circles
evaluation boolean, stop learning when the error starts to grow?
output string, what kind of model do you want? possible 'data' or 'space'
path_test string, path to testing file
space_limits list of lists of floats, minimal and maximal positions in space

list of outputs:
C numpy array kxd, matrix of k d-dimensional cluster centres
COV numpy array kxdxd, matrix of covariance matrices
density_integrals numpy array kx1, matrix of ratios between
                                   measurements and grid cells
                                   belonging to the clusters
structure list(int, list(floats), list(floats)),
          number of non-hypertime dimensions, list of hypertime
          radii nad list of wavelengths
average float, average value of measurements in cells
model numpy array (shape_of_grid), grid of modeled frequencies(stat)
(optional)
reality numpy array (shape_of_grid), grid of measured frequencies(stat)

NOTE: COV refers to precision matrices not to covariance matrices
      (it should be changed in all files)
"""

import os

import clustering as cl  # performs clustering of data
import dataset_io as dio  # loads and transforms data
import fremen as fm  # basic FreMEn to find most influential periodicity
import grid  # creates grid above data
import initialization as init  # initializes the learning proces (iteration 0)
import learning as lrn  # returns parameters of the learned model
import model as mdl  # returns model parameters and histogram above time-space
import testing as tst  # returns model (a grid of frequencies(stat))
###########################
# only during developement
# import importlib
# importlib.reload(cl)
# importlib.reload(dio)
# importlib.reload(fm)
# importlib.reload(grid)
# importlib.reload(init)
# importlib.reload(lrn)
# importlib.reload(mdl)
# importlib.reload(tst)
##########################

#############################
# people walking in corridors
#############################

# model cretion (two weeks)
directory = dio.file_directory()  # the directory of the file dataset_io.py


train_file_name = 'training_two_weeks.txt'
path_train = os.path.join(directory, '..', 'data', train_file_name)

# setting longest and shortest:
# longest - float, legth of the longest wanted period in default units,
#         - usualy four weeks
# shortest - float, legth of the shortest wanted period in default units,
#          - usualy one hour.
# It is necessary to understand what periodicities you are looking for (or what
#     periodicities you think are the most influential)
longest = 60*60*24*28
shortest = 60*60*12

# setting edge_of_square and timestep:
# timestep and edge_of_square has to be chosen based on desired granularity,
# timestep refers to the time variable,
# edge_of_square refers to other variables - it is supposed that the step
#     (edge of cell) in every variable other than time is equal.
#     If there are no other variables, some value has to be added but it is not
#     used.
edge_of_square = 0.5
timestep = 60*60*4

# setting k:
# k is number of clusters
# it looks like it is not necessary to have large number of clusters
k = 9

# setting radius:
# it is radius of the first created circle of the hyperspace
# based on chosen data, it looks like it is good for model, when the radius
# for hypertime circle created for one day periodicity is of size 2 (dunno Y)
# for other periodicities we can derived radius as
# radius = 2 * 86400 / another_periodicity
# so, if algorithm choose 604800 as a first (most influential) periodicity
# it is probably better to set this parameter to 2 * 86400 / 604800 = 0.2857
radius = 2

# setting number_of_periods and evaluation:
# number_of_periods - upper bound for a number of iterations of learning
# it is usualy lower than 5 on these data to add such hypertime that lower the
# precision of the model
# evaluation - if True, the learning stops when adding new hypertime lowers the
#              precision of the model and model without this hypertime is
#              returned
number_of_periods = 5
evaluation = True

C, COV, density_integrals, structure, average =\
    lrn.proposed_method(longest, shortest, path_train, edge_of_square,
                        timestep, k, radius, number_of_periods, evaluation)

# now, the parameters of the model for chosen grid are created
# and we will save them
saving_path = os.path.join(directory, '..', 'out')
# save C, numpy array kxd, matrix of k d-dimensional cluster centres
dio.save_numpy_array(variable=C, name="C", save_directory=saving_path)
# save COV, numpy array kxdxd, matrix of covariance matrices 
dio.save_numpy_array(variable=COV, name="COV", save_directory=saving_path)
# save density_integrals numpy array kx1, matrix of ratios between
#    measurements and grid cells belonging to the clusters
dio.save_numpy_array(variable=density_integrals, name="density_integrals",
                     save_directory=saving_path)
# save structure list(int, list(floats), list(floats)),
#    number of non-hypertime dimensions, 
#    list of hypertime radii nad list of wavelengths
dio.save_list(variable=structure, name="structure",
              save_directory=saving_path) 


# let us return the model - grid of frequencies(in the statistical meaning)

# setting output:
# we can create model above data to compare model and measured data
# (to do that , use output = 'data' )
# or we can create model above arbitrary space (of the same dimension as
# training dataset)
# (to do that , use output = 'space' )
output = 'space'
if output == 'data':
    # testing (two days)
    # parameters
    test_file_name = 'testing_two_days.txt'  # two days out of training data
    visualise = False  # if True it will  create lot of figures in ../fig/
    prefix = 'people_walking_testing_'  # to differentiate figures of models
    # call
    path_test = os.path.join(directory, '..', 'data', test_file_name)
    model, reality =\
        tst.model_above_data(path_test, C, COV, density_integrals,
                             structure, k, edge_of_square, timestep,
                             prefix, visualise, average)
elif output == 'space':
    # parameters (derived from the file testing_two_days.txt)
    # next lines show how to create space_limits
    min_t = 1496793601.0
    max_t = 1496966399.0
    min_x = -8.0
    max_x = 12.0
    min_y = -3.0
    max_y = 17.0
    space_limits = [[min_t, min_x, min_y], [max_t, max_x, max_y]]
    # call
    model =\
        tst.model_above_space(C, COV, density_integrals, structure, k,
                              edge_of_square, timestep,
                              space_limits)
    print('model created')


###############################################################################
# it was also tested on the data with different dimensionality and different
# interpretation of a model. Next lines are easy to start exploring...
#############################
# thoroughfare_times
#############################


## model creation (few months)
#directory = dio.file_directory()  # the directory of the file dataset_io.py
#
#
#train_file_name = 'thoroughfare_times_train.txt'
#path_train = os.path.join(directory, '..', 'data', train_file_name)
#
#longest = 60*60*24*7
#shortest = 60*60*2
#edge_of_square = 1
#timestep = 60*60
#k = 9
#radius = 2
#number_of_periods = 5
#evaluation = True
#
#C, COV, density_integrals, structure, average =\
#    lrn.proposed_method(longest, shortest, path_train, edge_of_square,
#                        timestep, k, radius, number_of_periods, evaluation)
#
#
#output = 'data'
##output = 'space'  # never tested, probably working ;)
#if output == 'data':
#    # testing (two weeks)
#    # parameters
#    test_file_name = 'thoroughfare_times_test.txt'
#    prefix = 'thoroughfare_times_testing'
#    visualise = True
#    # call
#    path_test = os.path.join(directory, '..', 'data', test_file_name)
#    model, reality =\
#        tst.model_above_data(path_test, C, COV, density_integrals,
#                             structure, k, edge_of_square, timestep,
#                             prefix, visualise, average)
