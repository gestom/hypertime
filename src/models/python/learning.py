# Created on Sun Aug 27 14:40:40 2017
# @author: tom

"""
returns parameters of the learned model
call proposed_method(longest, shortest, path, edge_of_square, timestep, k,
                     radius, number_of_periods, evaluation)
where
input: longest float, legth of the longest wanted period in default
                      units
       shortest float, legth of the shortest wanted period
                       in default units
       path string, path to file
       edge_of_square float, spatial edge of cell in default units (meters)
       timestep float, time edge of cell in default units (seconds)
       k positive integer, number of clusters
       radius float, size of radius of the first found hypertime circle
       number_of_periods int, max number of added hypertime circles
       evaluation boolean, stop learning when the error starts to grow?
and
output: C numpy array kxd, matrix of k d-dimensional cluster centres
        COV numpy array kxdxd, matrix of covariance matrices
        density_integrals numpy array kx1, matrix of ratios between
                                           measurements and grid cells
                                           belonging to the clusters
        structure list(int, list(floats), list(floats)),
                  number of non-hypertime dimensions, list of hypertime
                  radii nad list of wavelengths
        average DODELAT
"""

import numpy as np
from time import clock
import copy as cp

import model as mdl
import fremen as fm
import initialization as init
import dataset_io as dio
import evaluation as ev

def proposed_method(longest, shortest, dataset, edges_of_cell, k,
                    radius, number_of_periods, evaluation):
    """
    input: longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
           dataset numpy array, columns: time, vector of measurements, 0/1
                                (occurence of event)
           edge_of_square float, spatial edge of cell in default units (meters)
           timestep float, time edge of cell in default units (seconds)
           k positive integer, number of clusters
           radius float, size of radius of the first found hypertime circle
           number_of_periods int, max number of added hypertime circles
           evaluation boolean, stop learning when the error starts to grow?
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            COV numpy array kxdxd, matrix of covariance matrices
            density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
            structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
            average DODELAT
    uses: time.clock()
          init.whole_initialization(), iteration_step()
    objective: to learn model parameters
    """
    # initialization
    training_data, evaluation_dataset, training_dataset =\
        dio.divide_dataset(dataset)
    input_coordinates, overall_sum, structure, C, U,\
        shape_of_grid, time_frame_sums, T, W, ES, P, COV,\
        density_integrals, valid_timesteps =\
        init.whole_initialization(training_data, k, edges_of_cell,
                                  longest, shortest, training_dataset)
    # initialization of fiff, probably better inside  "whole_initialization"
    diff = -1
    # iteration
    if len(structure[1]) >= number_of_periods:
        jump_out = 1
    else:
        jump_out = 0
    iteration = 0
    if (P == 0.0 and structure[0] == 0) or (structure[0] and jump_out == 1):
        average = overall_sum / len(input_coordinates)                          
        C = np.array([average])                                                 
        COV = C/10                                                              
        density_integrals = np.array([[average]])                               
        structure =  [0, [], []]                                                
        jump_out = 1                                                          
        k = 1
    else:
        #print('trying to remove : Inf')
        try:
            WW = list(W)
            WW.remove(0.0)  # P
            W = np.array(WW)
            #print('frequency Inf removed')
        except ValueError:
            pass
    #elif P == 0.0:
    #    print(structure)
    #    jump_out, k, structure, C, U, COV, density_integrals, W, ES, P, diff=\
    #        step_evaluation(training_data, input_coordinates, structure,
    #                        C, U, k, shape_of_grid, time_frame_sums, T, W,
    #                        ES, COV, density_integrals, P, radius,
    #                        valid_timesteps,
    #                        evaluation_dataset, edges_of_cell, diff)
    while jump_out == 0:
        print('\nstarting learning iteration: ' + str(iteration))
        print('trying to remove chosen peridicity: ' + str(P))
        try:
            WW = list(W)
            WW.remove(1/P)  # P
            W = np.array(WW)
            print('periodicity ' + str(P) + ' removed')
        except ValueError:
            pass
        iteration += 1
        start = clock()
        if evaluation:
            jump_out, k, structure, C, U, COV, density_integrals, W, ES, P, diff=\
                step_evaluation(training_data, input_coordinates, structure,
                                C, U, k, shape_of_grid, time_frame_sums, T, W,
                                ES, COV, density_integrals, P, radius,
                                valid_timesteps,
                                evaluation_dataset, edges_of_cell, diff)
        else:
            structure[2].append(P)
            if len(structure[2]) == 1:
                structure[1].append(radius)
            else:
                #structure[1].append(structure[1][-1] *
                #                    structure[2][-2] / structure[2][-1])
                structure[1].append(radius)
            sum_of_amplitudes, C, U, COV, density_integrals, W, ES, P, diff =\
                iteration_step(training_data, input_coordinates, structure, C,
                               U, k, shape_of_grid, time_frame_sums, T, W, ES,
                               valid_timesteps,
                               evaluation_dataset, edges_of_cell)
            jump_out = 0
        if len(structure[1]) >= number_of_periods:
            jump_out = 1
        finish = clock()
        print('structure: ' + str(structure) + ' and number of clusters: ' + str(k))
        print('leaving learning iteration: ' + str(iteration))
        print('processor time: ' + str(finish - start))
    print('learning iterations finished')
    print('and the difference between model and reality at the end is:')
    if structure[0] == 0 and len(structure[1]) == 0:
        print('unknown, return average: ' + str(C[0]))
    else:
        if evaluation:
            list_of_diffs = []                                                       
            list_of_others = []                                                     
            for j in xrange(6):  # looking for the best clusters      
                sum_of_amplitudes_j, Cj, Uj, COVj, density_integrals_j, Wj, ESj,\
                    Pj, diff_j = iteration_step(training_data, input_coordinates,   
                                     structure, C, U, k, shape_of_grid,       
                                     time_frame_sums, T, W, ES, valid_timesteps,    
                                     evaluation_dataset, edges_of_cell)             
                list_of_diffs.append(diff_j)                            
                list_of_others.append((Cj, COVj, density_integrals_j))                           
            best_position = np.argmin(list_of_diffs)
            diff = list_of_diffs[best_position]
            C, COV, density_integrals = list_of_others[best_position]
            print('all diffs in comparison: ' + str(list_of_diffs))
        else:
            diff = ev.evaluation_step(evaluation_dataset, C, COV, density_integrals,\
                                      structure, k, edges_of_cell)
    print(diff)
    print('using k = ' + str(k))
    print('and structure: ' + str(structure) + '\n\n')
    average = overall_sum / len(input_coordinates)
    return C, COV, density_integrals, structure, average, k


def step_evaluation(training_data, input_coordinates, structure, C, U, k,
                    shape_of_grid, time_frame_sums, T, W, ES, COV,
                    density_integrals, P, radius, valid_timesteps,
                    evaluation_dataset, edges_of_cell, diff_old):
    """
    input: path string, path to file
           input_coordinates numpy array, coordinates for model creation
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           C numpy array kxd, centres from last iteration
           U numpy array kxn, matrix of weights from the last iteration
           k positive integer, number of clusters
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
           P float64, length of the most influential frequency in default
                      units
           radius float, size of radius of the first found hypertime circle
    output: jump_out int, zero or one - to jump or not to jump out of learning
            structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
            C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            COV numpy array kxdxd, matrix of covariance matrices
            density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
            W numpy array Lx1, sequence of reasonable frequencies
            ES float64, squared sum of squares of residues from this iteration
            P float64, length of the most influential frequency in default
                       units
    uses: iteration_step()
          cp.deepcopy()
    objective: to send new or previous version of model (and finishing pattern)
    """
    new_structure = cp.deepcopy(structure)
    if P > 0.0:
        new_structure[2].append(P)
        if len(new_structure[2]) == 1:
            new_structure[1].append(radius)
        else:
            #new_structure[1].append(new_structure[1][-1] *
            #                        new_structure[2][-2] / new_structure[2][-1])
            new_structure[1].append(radius)
##########################################
    last_best = ()
    sum_of_amplitudes = -1
    k_j = k
    all_params = []
    all_diffs = []
    while True:
        #k = k + 1
        list_of_sums = []
        list_of_others = []
        for j in xrange(3):  # for the case that the clustering would fail
            sum_of_amplitudes_j, Cj, Uj, COVj, density_integrals_j, Wj, ESj,\
                Pj, diff_j = iteration_step(training_data, input_coordinates,
                                 new_structure, C, U, k_j, shape_of_grid,
                                 time_frame_sums, T, W, ES, valid_timesteps,
                                 evaluation_dataset, edges_of_cell)
            list_of_sums.append(sum_of_amplitudes_j)
            list_of_others.append((Cj, Uj, COVj, density_integrals_j, Pj, Wj,
                                   ESj, k_j, diff_j))
            all_diffs.append(diff_j)
        all_params.append(list_of_others)
        chosen_model = np.argmin(list_of_sums)
        tested_sum_of_amplitudes = list_of_sums[chosen_model]
        if sum_of_amplitudes == -1:
            sum_of_amplitudes = tested_sum_of_amplitudes
            last_best = list_of_others[chosen_model]
            k_j = k_j + 1
        else:
            if tested_sum_of_amplitudes < sum_of_amplitudes:
                sum_of_amplitudes = tested_sum_of_amplitudes
                last_best = list_of_others[chosen_model]
                k_j = k_j + 1
            else:
                break
##########################################
    if last_best[-1] < diff_old or diff_old == -1:  # (==) if diff < diff_old
        C, U, COV, density_integrals, P, W, ES, k, diff = last_best
        structure = cp.deepcopy(new_structure)
        jump_out = 0
    else:
        jump_out = 1
        diff = diff_old
        print('\ntoo many periodicities, error have risen,')
        print('when structure ' + str(new_structure) + ' tested;')
        print('jumping out (prematurely)\n')
    return jump_out, k, structure, C, U, COV, density_integrals, W, ES, P, diff


def iteration_step(training_data, input_coordinates, structure, C_old, U_old,
                   k, shape_of_grid, time_frame_sums, T, W, ES,
                   valid_timesteps, evaluation_dataset, edges_of_cell):
    """
    input: path string, path to file
           input_coordinates numpy array, coordinates for model creation
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           C_old numpy array kxd, centres from last iteration
           U_old numpy array kxn, matrix of weights from the last iteration
           k positive integer, number of clusters
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
           time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
           T numpy array shape_of_grid[0]x1, time positions of timeframes
           W numpy array Lx1, sequence of reasonable frequencies
           ES float64, squared sum of squares of residues from this iteration
    output: dES float64, difference between last and new error
            structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
            C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            COV numpy array kxdxd, matrix of covariance matrices
            density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
            W numpy array Lx1, sequence of reasonable frequencies
            ES float64, squared sum of squares of residues from this iteration
            P float64, length of the most influential frequency in default
                       units
    uses: mdl.model_creation(), fm.chosen_period()
          np.sum()
    objective:
    """
    #### testuji zmenu "sily" period pri pridavani shluku
    hist_freqs, C, U, COV, density_integrals =\
        mdl.model_creation(input_coordinates,
                           structure, training_data, C_old, U_old, k,
                           shape_of_grid)
    osy = tuple(np.arange(len(np.shape(hist_freqs)) - 1) + 1)
    time_frame_freqs = np.sum(hist_freqs, axis=osy)
    P, W, ES, sum_of_amplitudes = fm.chosen_period(T, time_frame_sums,
                                                   time_frame_freqs, W, ES,
                                                   valid_timesteps)
    diff = ev.evaluation_step(evaluation_dataset, C, COV, density_integrals,\
                                      structure, k, edges_of_cell)
    #### konec testovani
    #print('chosen k: ' + str(k))
    #print('and the diff: ' + str(diff))
    return sum_of_amplitudes, C, U, COV, density_integrals, W,\
        ES, P, diff
