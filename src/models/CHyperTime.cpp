#include "CHyperTime.h"

using namespace std;

CHyperTime::CHyperTime(int id)
{
	type = TT_HYPER;
	modelPositive = modelNegative = NULL;
	spaceDimension = 1;
	timeDimension = 0;
	maxTimeDimension = 10;
	covarianceType = EM::COV_MAT_GENERIC;
	positives = negatives = 0;
	corrective = 1.0;
}

void CHyperTime::init(int iMaxPeriod,int elements,int numClasses)
{
	maxPeriod = iMaxPeriod;
	numElements = elements;
}

CHyperTime::~CHyperTime()
{
}

// adds new state observations at given times
int CHyperTime::add(uint32_t time,float state)
{
	sampleArray[numSamples].t = time;
	sampleArray[numSamples].v = state;
	numSamples++;
	return 0; 
}

/*required in incremental version*/
void CHyperTime::update(int modelOrder,unsigned int* times,float* signal,int length)
{
	int numTraining = numSamples;
	int numEvaluation = 0;
	if (order != modelOrder){
		delete modelPositive;
		delete modelNegative;
		modelPositive = modelNegative = NULL;
		order = modelOrder;
	}
	if (modelPositive == NULL) modelPositive = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
	if (modelNegative == NULL) modelNegative = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));

	/*separate positives and negative examples*/
	Mat samplesPositive(positives,spaceDimension+timeDimension,CV_32FC1);
	Mat samplesNegative(negatives,spaceDimension+timeDimension,CV_32FC1);
	
	float vDummy = 0.5;
	long int tDummy = 0.5;
	for (int i = 0;i<numTraining;i++){
		vDummy = sampleArray[i].v;
		if (vDummy > 0.5){
			samplesPositive.push_back(vDummy);
			positives++;
		} else{
			samplesNegative.push_back(vDummy);
			negatives++;
		}
	}
	periods.clear();
	bool stop = false;
	do { 
		/*find the gaussian mixtures*/
		if (positives <= order || negatives <= order)break;
		modelPositive->train(samplesPositive);
		modelNegative->train(samplesNegative);
		printf("Model trained with %i clusters, %i dimensions, %i positives and %i negatives\n",order,timeDimension,positives,negatives);
		/*analyse model error for periodicities*/
		CFrelement fremen(0);
		float err = 0;
		float sumErr = 0;
		fremen.init(maxPeriod,maxTimeDimension,1);

		/*calculate model error across time*/
		for (int i = 0;i<numTraining;i++)
		{
			fremen.add(sampleArray[i].t,estimate(sampleArray[i].t)-sampleArray[i].v);
		}

		/*determine model weights*/
		float integral = 0;
		numEvaluation = numSamples;
		for (int i = 0;i<numEvaluation;i++) integral+=estimate(sampleArray[i].t);
		corrective = corrective*positives/integral;
		
		/*calculate evaluation error*/
		for (int i = 0;i<numEvaluation;i++)
		{
			err = estimate(sampleArray[i].t)-sampleArray[i].v;
			sumErr+=err*err;
		}
		sumErr=sqrt(sumErr/numEvaluation);

		/*retrieve dominant error period*/	
		int maxOrder = 1;
		fremen.update(timeDimension/2+1);
		int period = fremen.predictFrelements[0].period;
		bool expand = true;
		fremen.print(true);
		printf("Model error with %i time dimensions and %i clusters is %.3f\n",timeDimension,order,sumErr);

		/*if the period already exists, then skip it*/
		for (int d = 0;d<timeDimension/2;d++)
		{
			if (period == periods[d]) period = fremen.predictFrelements[d+1].period;
		}
		errors[timeDimension/2] = sumErr;
		//cout << samplesPositive.rowRange(0, 1) << endl;

		/*error has increased: cleanup and stop*/
		if (timeDimension > 1 && errors[timeDimension/2-1] <  sumErr)
		{
			printf("Error increased from %.3f to %.3f\n",errors[timeDimension/2-1],errors[timeDimension/2]);
			timeDimension-=2;
			load("model");
			samplesPositive = samplesPositive.colRange(0, samplesPositive.cols-2);
			samplesNegative = samplesNegative.colRange(0, samplesNegative.cols-2);
			if (order < maxOrder){ 
				delete modelPositive;
				delete modelNegative;  
				modelPositive = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
				modelNegative = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
			}
			printf("Reducing hypertime dimension to %i: ",timeDimension);
			for (int i = 0;i<timeDimension/2;i++) printf(" %i,",periods[i]);
			printf("\n");
			stop = true;
		}else{
			save("model");
		}
		if (timeDimension >= maxTimeDimension) stop = true;

		/*hypertime expansion*/
		if (stop == false && expand == true){
			printf("Adding period %i \n",period);
			Mat hypertimePositive(positives,2,CV_32FC1);
			Mat hypertimeNegative(negatives,2,CV_32FC1);
			positives = negatives = 0;
			for (int i = 0;i<numTraining;i++)
			{
				vDummy = sampleArray[i].v;
				tDummy = sampleArray[i].t;
				if (vDummy > 0.5){
					hypertimePositive.at<float>(positives,0)=cos((float)tDummy/period*2*M_PI);
					hypertimePositive.at<float>(positives,1)=sin((float)tDummy/period*2*M_PI);
					positives++;
				}else{
					hypertimeNegative.at<float>(negatives,0)=cos((float)tDummy/period*2*M_PI);
					hypertimeNegative.at<float>(negatives,1)=sin((float)tDummy/period*2*M_PI);
					negatives++;
				}
			}
			hconcat(samplesPositive, hypertimePositive,samplesPositive);
			hconcat(samplesNegative, hypertimeNegative,samplesNegative);
			periods.push_back(period);
			timeDimension+=2;
		}
		if (order <  maxOrder) stop = false;
	}while (stop == false);
}

float CHyperTime::estimate(uint32_t t)
{
	/*is the model valid?*/
	if (modelNegative->isTrained() && modelPositive->isTrained()){
		Mat sample(1,spaceDimension+timeDimension,CV_32FC1);
		sample.at<float>(0,0)=1;

		/*augment data sample with hypertime dimensions)*/
		for (int i = 0;i<timeDimension/2;i++){
			sample.at<float>(0,spaceDimension+2*i+0)=cos((float)t/periods[i]*2*M_PI);
			sample.at<float>(0,spaceDimension+2*i+1)=sin((float)t/periods[i]*2*M_PI);
		}
		Mat probs(1,2,CV_32FC1);
		Vec2f a = modelPositive->predict(sample, probs);
//		cout << a << endl;
		sample.at<float>(0,0)=0;
		Vec2f b = modelNegative->predict(sample, probs);

		double d = ((positives*exp(a(0))+negatives*exp(b(0))));
		//d = ((exp(a(0))+exp(b(0))));
		//if(d > 0) return exp(a(0))/d;
		if(d > 0) return corrective*positives*exp(a(0))/d;
//		double d = ((exp(a(0))+exp(b(0))));
//		if(d > 0) return exp(a(0))/d;
	}
	/*any data available?*/
	if (negatives+positives > 0) return (float)positives/(positives+negatives);
	return 0.5;
}

void CHyperTime::print(bool all)
{
	Mat meansPositive = modelPositive->get<Mat>("means");
	Mat meansNegative = modelNegative->get<Mat>("means");
	std::cout << meansPositive << std::endl;
	std::cout << meansNegative << std::endl;	
	//std::cout << periods << std::endl;	
	printf("%i %i\n",order,timeDimension);	
}

float CHyperTime::predict(uint32_t time)
{
	return estimate(time);	
}

int CHyperTime::save(const char* name,bool lossy)
{
	/*EM models have to be saved separately and adding a '.' to the filename causes the saving to fail*/
	char filename[strlen(name)+5];

	sprintf(filename,"%spos",name);
	FileStorage fsp(filename, FileStorage::WRITE);
	fsp << "periods" << periods;
	fsp << "order" << order;
	fsp << "positives" << positives;
	fsp << "negatives" << negatives;
	fsp << "corrective" << corrective;
	if (modelPositive->isTrained()) modelPositive->write(fsp); 
	fsp.release();

	sprintf(filename,"%sneg",name);
	fsp.open(filename, FileStorage::WRITE);
	if (modelNegative->isTrained())	modelNegative->write(fsp); 
	fsp.release();

	return 0;
}

int CHyperTime::load(const char* name)
{
	/*EM models have to be saved separately and adding a '.' to the filename causes the saving to fail*/
	char filename[strlen(name)+5];

	sprintf(filename,"%spos",name);
	FileStorage fs(filename, FileStorage::READ);
	fs["periods"] >> periods;
	fs["order"] >> order;
	fs["positives"] >> positives;
	fs["negatives"] >> negatives;
	fs["corrective"] >> corrective;

	delete modelPositive;
	delete modelNegative;
	modelPositive = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
	modelNegative = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));

	FileNode fn = fs["StatModel.EM"];
	timeDimension = periods.size()*2;
	modelPositive->read(fn); 
	fs.release();

	sprintf(filename,"%sneg",name);
	fs.open(filename, FileStorage::READ);
	fn = fs["StatModel.EM"];
	modelNegative->read(fn); 
	fs.release();

	return 0;
}

int CHyperTime::save(FILE* file,bool lossy)
{
//	int frk = numElements;
//	fwrite(&frk,sizeof(uint32_t),1,file);
//	fwrite(&storedGain,sizeof(float),1,file);
//	fwrite(storedFrelements,sizeof(SFrelement),numElements,file);
	return 0;
}

int CHyperTime::load(FILE* file)
{
//	int frk = numElements;
//	fwrite(&frk,sizeof(uint32_t),1,file);
//	fwrite(&storedGain,sizeof(float),1,file);
//	fwrite(storedFrelements,sizeof(SFrelement),numElements,file);
	return 0;
}

/*this is very DIRTY, but I don't see any other way*/
int CHyperTime::exportToArray(double* array,int maxLen)
{
	save("hypertime.tmp");
	array[0] = TT_HYPER;
	if (modelNegative->isTrained() && modelPositive->isTrained()){
		FILE*  file = fopen("hypertime.tmpneg","r");
		int len = fread(&array[5],1,maxLen,file);
		fclose(file);	
		array[1] = len;
		file = fopen("hypertime.tmppos","r");
		len = fread(&array[len+5],1,maxLen,file);
		array[2] = len;
		fclose(file);	
		return array[1]+array[2]+5;
	}else{
		array[1] = 0;
		array[2] = positives;
		array[3] = negatives;
		array[4] = order;
		return 5;
	}
}

/*this is very DIRTY, but I don't see any other way*/
int CHyperTime::importFromArray(double* array,int len)
{
	if (array[1] > 0){
		FILE*  file = fopen("hypertime.tmpneg","w");
		fwrite(&array[5],1,array[1],file);
		fclose(file);
		file = fopen("hypertime.tmppos","w");
		fwrite(&array[(int)array[1]+5],1,array[2],file);
		fclose(file);
		load("hypertime.tmp");
	}else{
		periods.clear();
		positives = array[2];
		negatives = array[3];
		order = array[4];
		if (modelPositive == NULL) modelPositive = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
		if (modelNegative == NULL) modelNegative = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
	}
	return 0;
}
