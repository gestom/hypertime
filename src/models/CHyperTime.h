#ifndef CHYPERTIME_H
#define CHYPERTIME_H

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
#include "CFrelement.h"


/**
@author Tom Krajnik
*/
typedef struct{
	long int t;
	float v;
}SHyperTimeSample;

using namespace std;

class CHyperTime: public CTemporal
{
	public:
		CHyperTime(int id);
		~CHyperTime();

		//adds a serie of measurements to the data
		int add(uint32_t time,float state);
		void init(int iMaxPeriod,int elements,int numClasses);

		//estimates the probability for the given times 
		float estimate(uint32_t time);
		float predict(uint32_t time);

		void update(int maxOrder,unsigned int* times = NULL,float* signal = NULL,int length = 0);
		void print(bool verbose=true);

		int exportToArray(double* array,int maxLen);
		int importFromArray(double* array,int len);
		int save(FILE* file,bool lossy = false);
		int load(FILE* file);
		int save(const  char* name,bool lossy = false);
		int load(const  char* name);
		
		SHyperTimeSample sampleArray[1000000];
		int numSamples;
		int positives;
		int negatives;
		EM* modelPositive;
		EM* modelNegative;
		int spaceDimension;
		int timeDimension;
		int covarianceType;
		vector<int> periods;
		float errors[100];
		float corrective;
		int maxTimeDimension;
};

#endif
