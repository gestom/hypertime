import numpy as np
import learning as lrn
import model as mdl

def python_function_update(dataset):
    """
    input: training_coordinates numpy array nxd, measured values in measured
                                                 times
    output: probably whole model
    uses: 
    objective: to call warpHypertime and return all parameters of the found
               model
    """
    ###################################################
    # otevirani a zavirani dveri, pozitivni i negativni
    ###################################################
    # differentiate to positives and negatives
    # path_n = training_coordinates[training_coordinates[:,1] == 0][:, 0:1]
    # path_p = training_coordinates[training_coordinates[:,1] == 1][:, 0:1]
    # training_coordinates = None  # free memory?
    # parameters
    #### testovani zmeny "sily" periody pri zmene poctu shluku
    longest = 60*60*24*7*4 # testing one day
    shortest = 60*60*4 # testing one day
    #### konec testovani
    edges_of_cell = [60]
    k = 1  # muzeme zkusit i 9
    # hours_of_measurement = 24 * 7  # nepotrebne
    radius = 1.0
    number_of_periods = 4
    evaluation = True 
    C_p, COV_p, density_integrals_p, structure_p, average_p, k_p =\
        lrn.proposed_method(longest, shortest, dataset,
                            edges_of_cell, k,
                            radius, number_of_periods, evaluation)
    return C_p, COV_p, density_integrals_p, structure_p, k_p


def python_function_estimate(whole_model, time):
    """
    input: whole_model tuple of model parameters, specificaly:
                C_p, COV_p, densities_p, structure_p, k_p
           time float, time for prediction
    output: estimation float, estimation of the event occurence
    uses:
    objective: to estimate event occurences in the given time
    """
    ###################################################
    # otevirani a zavirani dveri, pozitivni i negativni
    ###################################################
    if whole_model[3][0] == 0 and len(whole_model[3][1]) == 0:  # no model
        return whole_model[0][0]  # average
    else:
        freq_p = mdl.one_freq(np.array([[time]]), whole_model[0],
                              whole_model[1], whole_model[3], whole_model[4],
                              whole_model[2])
    return float(freq_p[0])


def python_function_save(whole_model, file_path):
    """
    """
    #with open(file_path, 'wb') as opened_file:
    #    np.savez(opened_file, whole_model[0], whole_model[1], whole_model[2],
    #             whole_model[3], whole_model[4])
    structure = whole_model[3]
    dim_hyp = len(structure[1])
    new_structure = [structure[0]]
    for i in xrange(2):
        for j in xrange(dim_hyp):
            new_structure.append(structure[i + 1][j])
    structure_to_save = np.array(new_structure)
    np.savez(file_path, whole_model[0], whole_model[1], whole_model[2],
            structure_to_save, whole_model[4])


def python_function_load(file_path):
    """
    """
    with open(file_path + '.npz', 'r') as opened_file:
        npzfile = np.load(opened_file)
        C_p, COV_p, density_integrals_p, loaded_structure, k =\
            npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'],\
            npzfile['arr_3'], int(npzfile['arr_4'])
    dim_hyp = int((len(loaded_structure) - 1) / 2)
    structure = [int(loaded_structure[0])]
    for i in xrange(2):
        sub_list = []
        for j in xrange(dim_hyp):
            sub_list.append(loaded_structure[1 + i * dim_hyp + j])
        structure.append(sub_list)
    return C_p, COV_p, density_integrals_p, structure, k


def python_function_model_to_array(whole_model):
    """
    indian style :)
    """
    # C
    C_0 = whole_model[0]
    shape_C_0 = np.shape(C_0)
    reshaped_C_0 = np.reshape(C_0, -1)
    len_shape_C_0 = len(shape_C_0)
    len_reshaped_C_0 = len(reshaped_C_0)
    # COV
    COV_1 = whole_model[1]
    shape_COV_1 = np.shape(COV_1)
    len_shape_COV_1 = len(shape_COV_1)
    reshaped_COV_1 = np.reshape(COV_1, -1)
    len_reshaped_COV_1 = len(reshaped_COV_1)
    # density_integrals DI
    DI_2 = whole_model[2]
    shape_DI_2 = np.shape(DI_2)
    len_shape_DI_2 = len(shape_DI_2)
    reshaped_DI_2 = np.reshape(DI_2, -1)
    len_reshaped_DI_2 = len(reshaped_DI_2)
    # structure S - different approach, list to array
    structure = whole_model[3]
    dim_hyp = len(structure[1])
    new_structure = [structure[0]]
    for i in xrange(2):
        for j in xrange(dim_hyp):
            new_structure.append(structure[i + 1][j])
    reshaped_S_3 = np.array(new_structure)
    shape_S_3 = np.shape(reshaped_S_3)
    len_shape_S_3 = len(shape_S_3)
    len_reshaped_S_3 = len(reshaped_S_3)
    # k - is number only
    k_4 = whole_model[4]
    shape_k_4 = 1
    len_shape_k_4 = 1
    reshaped_k_4 = np.array(k_4)
    len_reshaped_k_4 = 1
    # array creation
    length_of_array = 6 + len_shape_C_0 + len_shape_COV_1 + len_shape_DI_2 +\
                      len_shape_S_3 + len_shape_k_4 + len_reshaped_C_0 +\
                      len_reshaped_COV_1 + len_reshaped_DI_2 +\
                      len_reshaped_S_3 + len_reshaped_k_4
    output_array = np.empty(length_of_array)
    # filling
    output_array[0] = length_of_array
    output_array[1] = len_shape_C_0
    output_array[2] = len_shape_COV_1
    output_array[3] = len_shape_DI_2
    output_array[4] = len_shape_S_3
    output_array[5] = len_shape_k_4
    start = 5 + 1
    end = start + len_shape_C_0
    output_array[start: end] = np.array(shape_C_0)
    start = end
    end = start + len_shape_COV_1
    output_array[start: end] = np.array(shape_COV_1)
    start = end
    end = start + len_shape_DI_2
    output_array[start: end] = np.array(shape_DI_2)
    start = end
    end = start + len_shape_S_3
    output_array[start: end] = np.array(shape_S_3)
    start = end
    end = start + len_shape_k_4
    output_array[start: end] = np.array(shape_k_4)
    start = end
    end = start + len_reshaped_C_0
    output_array[start: end] =  reshaped_C_0
    start = end
    end = start + len_reshaped_COV_1
    output_array[start: end] =  reshaped_COV_1
    start = end
    end = start + len_reshaped_DI_2
    output_array[start: end] =  reshaped_DI_2
    start = end
    end = start + len_reshaped_S_3
    output_array[start: end] =  reshaped_S_3
    start = end
    end = start + len_reshaped_k_4
    output_array[start: end] =  reshaped_k_4
    return output_array


def python_function_array_to_model(input_array):
    """
    """
    number_of_parameters = 5
    len_shapes = []
    position = 0  # we do not need the zeroth position information
    for i in range(number_of_parameters):
        position += 1
        len_shapes.append(input_array[position])

    len_params = []
    shape_params = []
    for j in range(number_of_parameters):
        one_param_shape = []
        one_param_len = 1
        for k in range(int(len_shapes[j])):
            position += 1
            temp = input_array[position]
            one_param_len *= temp
            one_param_shape.append(temp)
        len_params.append(one_param_len)
        shape_params.append(one_param_shape)

    parameters = []
    end = position + 1
    for l in range(number_of_parameters):
        start = end
        end = start + int(len_params[l])
        parameters.append(np.reshape(input_array[start: end],
                                     np.int64(shape_params[l])))
        #parameters.append(input_array[start: end])

    # to use identical process as in load function
    # structure transformation
    loaded_structure = parameters[3]
    dim_hyp = (len(loaded_structure) - 1) / 2
    structure = [int(loaded_structure[0])]
    for i in xrange(2):
        sub_list = []
        for j in xrange(dim_hyp):
            sub_list.append(loaded_structure[1 + i * dim_hyp + j])
        structure.append(sub_list)

    # k transformation
    k = int(parameters[4])

    return parameters[0], parameters[1], parameters[2], structure, k
