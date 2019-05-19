#ifndef CPYTHONHYPERTIME_H
#define CPYTHONHYPERTIME_H

// tyto knihovny jsou potreba
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>


#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "CTimer.h"
#include "CTemporal.h"


#define FREMEN_AMPLITUDE_THRESHOLD 0.0
	
/**
@author Tom Krajnik
*/

using namespace std;

class CPythonHyperTime: public CTemporal
{
	public:
		CPythonHyperTime(int id);
		~CPythonHyperTime();

		//adds a serie of measurements to the data
		int add(uint32_t time,float state);
		void init(int iMaxPeriod,int elements,int numActivities);

		//estimates the probability for the given times 
		float estimate(uint32_t time);
		float predict(uint32_t time);

		void update(int maxOrder,unsigned int* times = NULL,float* signal = NULL,int length = 0);
		int exportToArray(double* array,int maxLen);
		int importFromArray(double* array,int len);
		void print(bool verbose=true);

		int save(FILE* file,bool lossy = false);
		int load(FILE* file);
		int save(const char* name,bool lossy = false);
		int load(const char* name);
		
		char id[MAX_ID_LENGTH];
		int measurements;

		PyObject *pModuleName;
		PyObject *pModule;
		PyObject *pModel;
        PyObject *pFunc2;

};

#endif
