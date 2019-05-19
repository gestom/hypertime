#include "CFrelement.h"
#include "CPerGaM.h"
#include "CTimeAdaptiveHist.h"
#include "CTimeHist.h"
#include "CTimeNone.h"
#include "CTimeMean.h"
#include "CMises.h"
#include "CPythonHyperTime.h"
#include "CHyperTime.h"

const char *temporalModelName[] = 
{
	"None",
	"Mean",
	"Hist",
	"FreMEn",
	"HyT-EM",
	"HyT-KM",
	"Gaussian",
	"Adaptive",
	"VonMises",
	"Number"
};


CTemporal* spawnTemporalModel(ETemporalType type,int maxPeriod,int elements,int numClasses)
{
	CTemporal *temporalModel;
	switch (type)
	{
		case TT_NONE: 		temporalModel = new CTimeNone(0);		break;
		case TT_MEAN: 		temporalModel = new CTimeMean(0);		break;
		case TT_HISTOGRAM:	temporalModel = new CTimeHist(0);		break;
		case TT_FREMEN: 	temporalModel = new CFrelement(0);		break;
		case TT_HYPER: 		temporalModel = new CHyperTime(0);		break;
		case TT_PYTHON: 	temporalModel = new CPythonHyperTime(0);	break;

		case TT_PERGAM: 	temporalModel = new CPerGaM(0);			break;
		case TT_ADAPTIVE: 	temporalModel = new CTimeAdaptiveHist(0);	break;
		case TT_MISES: 		temporalModel = new CMises(0);			break;
		default: 		temporalModel = new CTimeNone(0);
	}
	temporalModel->init(maxPeriod,elements,numClasses);
	return temporalModel;
}

CTemporal* spawnTemporalModel(const char* type,int maxPeriod,int elements,int numClasses)
{
	int i = TT_NONE;
	for (i=0;i<TT_NUMBER && strcmp(type,temporalModelName[i])!=0;i++){}
	return spawnTemporalModel( (ETemporalType)i,maxPeriod,elements,numClasses);
}
