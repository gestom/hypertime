#ifndef CMISES_H
#define CMISES_H

#include <opencv2/ml/ml.hpp>

using namespace cv;

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "CTimer.h"
#include "CTemporal.h"

/**
@author Tom Krajnik
*/
typedef struct{
	long int t;
	float x,y;
}SSample;

using namespace std;

class CMises: public CTemporal
{
	public:
		CMises(int id);
		~CMises();

		//adds a serie of measurements to the data
		int add(uint32_t time,float state);
		void init(int iMaxPeriod,int elements,int numActivities);

		//estimates the probability for the given times 
		float estimate(uint32_t time);
		float predict(uint32_t time);

		void update(int maxOrder,unsigned int* times = NULL,float* signal = NULL,int length = 0);
		void print(bool verbose=true);

		int exportToArray(double* array,int maxLen);
		int importFromArray(double* array,int len);
		int save(FILE* file,bool lossy = false);
		int load(FILE* file);
		int save(const char* name,bool lossy = false);
		int load(const char* name);
		
		SSample positiveArray[1000000];
		SSample negativeArray[1000000];
		int negatives,positives;
		EM* modelPositive;
		EM* modelNegative;
};

#endif
