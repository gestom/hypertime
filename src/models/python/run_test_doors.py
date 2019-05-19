#!/usr/bin/env python2


import numpy as np
#import python_module as pm
import tested_doors_python_module as pm
import dataset_io as dio

#c = dio.loading_data('../data/training_two_weeks_01.txt')
#c = dio.loading_data('../data/10_weeks_doors.txt')
c = dio.loading_data('../data/training_data.txt')


#a = np.array([0, 7200, 14400, 21600, 28800, 36000, 43200, 50400, 57600, 64800, 72000, 79200])
#b = np.array([0,0,1,1,0,0,1,1,0, 0,1, 1])
#c = np.c_[a, b]

"""
with open('../data/data.txt', 'r') as f:
    i = 0
    for line  in f:
        #print(line)
        b = np.array(map(float, list(line[1:-2].split(', '))))
        a = np.arange(len(b)) * 600
        #print('a: ' + str(np.shape(a)))
        #print('b: ' + str(np.shape(b)))
        c = np.c_[a, b]
        #print('c: ' + str(np.shape(c)))
        #print(c)
        i += 1
        model = pm.python_function_update(c)
        if i == 14:
            break
        else:
            print('feature no.: ' + str(i))

#print(list(b))
"""
model = pm.python_function_update(c)
print('update prosel')

pm.python_function_save(model, 'save_pokus')
print('save prosel')

model = pm.python_function_load('save_pokus')
print('load prosel')

out_array = pm.python_function_model_to_array(model)
print('array prosel')


model = pm.python_function_array_to_model(out_array)
print('model zrekonstruovan')


print(model[0])
print(model[1])
print(model[2])
print(model[3])
print(model[4])
#for i in xrange(10):
#    print(pm.python_function_estimate(model, c[-1, 0] + i * 60))



"""
C = np.arange(20).reshape((5, 4))
COV = np.arange(20).reshape((5, 2, 2))
DI = np.arange(5).reshape((5, 1))
structure = [0, [1.0, 1.0], [86400.0, 640000.0]]
k = 6
out_array = pm.python_function_model_to_array((C, COV, DI, structure, k))
out_model = pm.python_function_array_to_model(out_array)
print(C)
print(out_model[0])
print(COV)
print(out_model[1])
print(DI)
print(out_model[2])
print(structure)
print(out_model[3])
print(k)
print(out_model[4])
"""
