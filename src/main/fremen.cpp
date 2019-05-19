#include <iostream>
#include <fstream>	
#include <cstdlib>	
#include "CTemporal.h"
#include "CTimer.h"
#define MAX_SIGNAL_LENGTH 1000000

CTemporal *temporalModel;
int trainingTimes[MAX_SIGNAL_LENGTH];
unsigned char trainingStates[MAX_SIGNAL_LENGTH];
int testingTime;
float predictions[MAX_SIGNAL_LENGTH];

int testingLength = 0;
int trainingLength = 0;
int dummyState = 0;
int dummyTime = 0;

int main(int argc,char *argv[])
{
	/*read training data*/
	FILE* file=fopen(argv[1],"r");
	while (feof(file)==0)
	{
		fscanf(file,"%i %i\n",&dummyTime,&dummyState);
		trainingTimes[trainingLength] = dummyTime;
		trainingStates[trainingLength] = dummyState;
		trainingLength++;
	}
	fclose(file);

	/*traning model*/
	temporalModel = spawnTemporalModel(argv[3],60*60*24*7,atoi(argv[4]),1);

	if (atoi(argv[4])==0 && argv[4][0]!='0'){
		temporalModel->load(argv[4]);
	}else{
		for (int i = 0;i<trainingLength;i++){
			temporalModel->add(trainingTimes[i],trainingStates[i]);
		}
		temporalModel->update(atoi(argv[4]));
		temporalModel->save("model");
	}
	temporalModel->print(true);
	/*double array[100000];
	int len = temporalModel->exportToArray(array,100000);
	CTemporal* temporalModel2 = new CHyperTime(5);
	temporalModel2->init(86400,4,1);
	temporalModel2->importFromArray(array,len);
	temporalModel2->print(true);*/
		
	/*read testing timestamps and make predictions*/
	file=fopen(argv[2],"r");
	while (feof(file)==0){
		fscanf(file,"%i\n",&testingTime);
		predictions[testingLength++] = temporalModel->predict(testingTime);
	}
	fclose(file);

	file=fopen("predictions.txt","w");
	for (int i =0;i<testingLength;i++) fprintf(file,"%.3f\n",predictions[i]);
	fclose(file);
	return 0;
}
