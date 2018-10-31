/////////////////////////////////////////////////////////////////////
//
// Implements the NNetTrainer class
//
// Author: Jason Jenkins
//
// This class provides a framework for training a neural network.
//
// Once a network has been built it can then be trained using a
// suitable set of input and output values - known as the training 
// set.  Typically this set of values is quite large and for each 
// input element there is a corresponding output or target element.
// The training set input values are the inputs into the neural 
// network and the corresponding target values represents the 
// output that you would like the network to reproduce in response 
// to the given input.
// 
// As an example: if the network has three input units and two output
// units then each input element of the training set will have three 
// values (stored in a three element vector) and each target element 
// of the training set will have two values (stored in a two element
// vector).
//
// The calling process of this trainer must first add a suitable 
// training set consisting of the input and output values using the 
// addNewTrainingSet method. The training parameters - the learning 
// constant and the momentum - can then be optionally set using the 
// setLearningConstant and setMomentum methods respectively. If these
// parameters are not set then default values will be used instead. 
// The training routine can then be called by passing the neural 
// network that has been built with the required architecture to the
// trainNeuralNet method which will carry out the training. The 
// following code outlines this process:
/*
    // create the trainer
    NNetTrainer trainer;

    // the training set needs to be stored as follows
    vector<vector<double> > inputVectors;
    vector<vector<double> > targetVectors

    // a routine to populate the training set is required ...

    // initialize the trainer
    trainer.addNewTrainingSet(inputVectors, targetVectors);
    trainer.setLearningConstant(0.05);
    trainer.setMomentum(0.25);	

    // train the network - here Net is the prior built network
    trainer.trainNeuralNet(Net);

    // inspect the total network error
    double netError = trainer.getNetError();

    // if more training is required then reset the network error
    trainer.resetNetError();

    // and call the training routine again
    trainer.trainNeuralNet(net);
	netError = trainer.getNetError();

	// and repeat until training is complete
*/
//
// The training routine feeds the first input vector of the training
// set into the network. The output of the network - known as the 
// response - is calculated and compared to the ideal output in the 
// corresponding target vector of the training set. The difference 
// between the response and target values - known as error - is 
// calculated. The error value is then fed back through the network 
// using a procedure called Backpropagation - the Wikipedia entry 
// for this process can be found here:
//
//         https://en.wikipedia.org/wiki/Backpropagation. 
//
// The weighted connections linking the layers of the network 
// together (initially set to random values) are adjusted by this
// process using the gradient descent method to minimize the error. 
// The next input element of the training set is then fed into the 
// network and the response and associated error are calculated 
// again and used to further adjust the weighted connections. This
// process is continued until all the training set has been fed into
// the network and the errors occurring at each step are totalled
// together to give the total network error.
//
// The calling process of the trainer can then interrogate this 
// value and if the total network error is found to be less than a
// given predetermined value the training is deemed complete. But if
// the total error is still above the predetermined value the network
// error can be reset to zero and the training process can be called
// again and again until the total error has reached the desired 
// level or a set number of iterations has been exceeded.
//
/////////////////////////////////////////////////////////////////////

#include "NNetTrainer.h"

/////////////////////////////////////////////////////////////////////

#include <math.h>
#include <algorithm>
#include <iostream>

/////////////////////////////////////////////////////////////////////
// default constructor

NNetTrainer::NNetTrainer()
{
	mNetError = 0;

	// the default learning constant and momentum
	mLearnConst = 0.5;
	mMomentum = 0;
}

/////////////////////////////////////////////////////////////////////
// destructor

NNetTrainer::~NNetTrainer()
{
}

/////////////////////////////////////////////////////////////////////
// Public Methods
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// setLearningConstant - sets the learning constant training parameter
//
// The learning constant governs the 'size' of the steps taken down
// the error surface. Larger values decrease training time but can
// lead to the system overshooting the minimum value.
//
// learnConst (double) - the learning constant value

void NNetTrainer::setLearningConstant(double learnConst)
{ 
	// ignore invalid values
	if(learnConst > 0)
	{
		mLearnConst = learnConst;
	}
}

/////////////////////////////////////////////////////////////////////
// setMomentum - sets the momentum training parameter
//
// The momentum term can be used to weight the search of the error 
// surface to continue along the same 'direction' as the previous 
// step.
//
// A value of 1 will add 100% of the previous weighted connection
// value to the next weighted connection adjustment.  If set to 
// zero (the default value) the next step of the search will always
// proceed down the steepest path of the error surface.
// 
// momentum (double) - the momentum value

void NNetTrainer::setMomentum(double momentum)
{
	// ignore invalid values
	if(momentum > 0)
	{
		mMomentum = momentum;
	}
}

/////////////////////////////////////////////////////////////////////
// trainNeuralNet - trains the supplied neural network
//
// Each time this method is called the order of the training set
// elements are randomly shuffled to try and avoid any potential
// bias toward certain patterns that may occur if the data
// were always presented to the trainer in the same order.
//
// nNet (NeuralNet) - the neural network to be trained

void NNetTrainer::trainNeuralNet(NeuralNet& nNet)
{	
	int nTrain = (int)mTrainInput.size();
	
	vector<int> idx;

	// populate the index list for the training set
	for(int i = 0; i < nTrain; i++)
	{
		idx.push_back(i);
	}

	if(nTrain > 0)
	{
		// randomly shuffle the index list
		std::random_shuffle(idx.begin(), idx.end());

		for(int i = 0; i < nTrain; i++)
		{
			int index = idx[i];
			vector<double> outVec;              // the network output values
			vector<double> outErrSig;           // the output layer errors
			vector<vector<double> > hidErrSig;  // the hidden layer errors

			// get the next input values vector from the training set
			vector<double> trainVec = mTrainInput[index];

			// calculate the response from the training set input vector
			nNet.getResponse(trainVec, outVec);

			// calculate the total network error
			mNetError += calcNetworkError(outVec, index);

			// calculate the error signal on each output unit
			calcOutputError(nNet, outErrSig, outVec, index);

			// calculate the error signal on each hidden unit
			calcHiddenError(hidErrSig, outErrSig, nNet);

			// calculate the weight adjustments for the connections into the output layer
			calcOutputWtAdjust(outErrSig, nNet);

			// calculate the weight adjustments for the connections into the hidden layers
			calcHiddenWtAdjust(hidErrSig, trainVec, nNet);
		}
	}
}

/////////////////////////////////////////////////////////////////////
// addToTrainingSet - adds an individual input vector and 
//                    corresponding target vector to the training set
//
// inVec (vector<double>) - a vector of input values
// outVec (vector<double>) - a vector of target values

void NNetTrainer::addToTrainingSet(const vector<double>& inVec,
								   const vector<double>& outVec)
{
	mTrainInput.push_back(inVec);
	mTrainTarget.push_back(outVec);
}

/////////////////////////////////////////////////////////////////////
// addNewTrainingSet - adds a complete training set of input and 
//                     corresponding target vectors to the trainer
//
// A single input element of the training set consists of a vector of
// values hence the complete set of input values consists of a vector
// of vectors. The complete target set is similarly defined.
//
// inVecs (vector<vector<double>) - a vector of input vector values
// outVecs (vector<vector<double>) - a vector of target vector values

void NNetTrainer::addNewTrainingSet(const vector<vector<double> >& inVecs, 
									const vector<vector<double> >& outVecs)
{
	mTrainInput.clear();
	mTrainTarget.clear();

	mTrainInput = inVecs;
	mTrainTarget = outVecs;
}

/////////////////////////////////////////////////////////////////////
// Private Methods
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// calcNetworkError - calculates the network error between a given 
//                    vector of response values and the corresponding 
//                    vector of target values
//
// response (vector<double>) - the network response values
// nTarget (int) - the index of the corresponding target values in 
//                 the training set
//
// returns the current network error value

double NNetTrainer::calcNetworkError(const vector<double>& response, int nTarget)
{
	double error = 0;
	vector<double> targetVec = mTrainTarget[nTarget];

	for(int i = 0; i < (int)response.size(); i++)
	{
		error += 0.5 * pow((targetVec[i] - response[i]), 2);
	}

	return error;
}

/////////////////////////////////////////////////////////////////////
// calcOutputError - calculates the error signal on each individual
//                   unit in the output layer
//
// Uses the gradient descent method to search the error surface.
//
// outErr (vector<double>) - the calculated output unit errors
// response (vector<double>) - the network response values
// nTarget (int) - the index of the corresponding target values in 
//                 the training set

void NNetTrainer::calcOutputError(NeuralNet& nNet, vector<double>& outErr,
								  const vector<double>& response, int nTarget)
{
	vector<double> unitInputs, targetVec = mTrainTarget[nTarget];
	
	// get the output layer activation unit details
	ActiveT outType = nNet.getOutputUnitType();
	double outSlope = nNet.getOutputUnitSlope();
	double outAmplify = nNet.getOutputUnitAmplify();

	// get the output layer activation unit input values
	nNet.getUnitInputs(unitInputs, nNet.getNumLayers());

	for(int i = 0; i < (int)response.size(); i++)
	{
		double yi = response[i];
		double xi = unitInputs[i];

		// follow the steepest path on the error function by moving along the gradient
		// of the output units activation function - the gradient descent method
		double err = (targetVec[i] - yi) * getGradient(outType, outSlope, outAmplify, xi);

		outErr.push_back(err);
	}
}

/////////////////////////////////////////////////////////////////////
// calcHiddenError - calculates the error signal on each individual 
//                   unit within the networks hidden layers
//
// Uses the gradient descent method to search the error surface.
//
// hidErr (vector<vector<double> >) - the calculated hidden unit errors
// outErr (vector<double>) - the output unit errors
// nNet (NeuralNet) - the network undergoing training

void NNetTrainer::calcHiddenError(vector<vector<double> >& hidErr, 
	                              const vector<double>& outErr, NeuralNet& nNet)
{
	ActiveT unitType;
	double slope, amplify;
	int nHidden = nNet.getNumLayers();

	// initialise the the previous layer error with the output layer errors
	vector<double> prevErr = outErr;

	// start with the last hidden layer and work back to the first
	for(int i = nHidden; i >= 1; i--)
	{
		vector<double> unitInputs, activations, layerErr;

		// get the weighted connections for the current hidden layer
		NNetWeightedConnect wtConnect;		
		nNet.getWeightedConnect(wtConnect, i);
		int nUnits = wtConnect.getNumInputNodes();
		int nConnect = wtConnect.getNumOutputNodes();
				
		// get the hidden layer activation unit details
		nNet.getLayerDetails(i - 1, unitType, slope, amplify);

		// get the hidden layer activation unit input values
		nNet.getUnitInputs(unitInputs, i - 1);

		// calculate the hidden layer errors
		for(int j = 0; j < nUnits; j++)
		{
			double error = 0;
			double xj = unitInputs[j];
			
			for(int k = 0; k < nConnect; k++)
			{
				vector<double> weights;
				wtConnect.getWeightVector(k, weights);

				// follow the steepest path on the error function by moving along the gradient
				// of the hidden layer units activation function - the gradient descent method
				error += getGradient(unitType, slope, amplify, xj) * prevErr[k] * weights[j];
			}

			layerErr.push_back(error);
		}

		// update the hidden errors with the current layer error
		// N.B. Since we start from the last hidden layer the 
		// hidden layer error signals are stored in reverse order
		hidErr.push_back(layerErr);

		// back propagate the layer errors
		prevErr.clear();
		prevErr = layerErr;
	}
}

/////////////////////////////////////////////////////////////////////
// calcOutputWtAdjust - calculates the weight adjustments for the 
//                      connections into the output layer
//
// outErr (vector<double>) - the output unit errors
// nNet (NeuralNet) - the network undergoing training

void NNetTrainer::calcOutputWtAdjust(const vector<double>& outErr, NeuralNet& nNet)
{
	vector<double> xVec;
	int n = nNet.getNumLayers(), prevIdx = 0;

	// get the weighted connections between the last hidden layer and the output layer
	NNetWeightedConnect wtConnect;		
	nNet.getWeightedConnect(wtConnect, n);
	
	// get the input values for the weighted connections
	nNet.getActivations(xVec, n - 1);

	int nOut = wtConnect.getNumOutputNodes();

	// calculate the weight adjustments for each weighted connection output unit
	for(int i = 0; i < nOut; i++)
	{
		double ei = outErr[i];
		vector<double> weights;

		// get the output units weight vector
		wtConnect.getWeightVector(i, weights);

		// calculate the total weight adjustment
		for(int j = 0; j < (int)xVec.size(); j++)
		{
			double dWPrev = 0;

			// the weight adjustment calculation
			double dW = mLearnConst * ei * xVec[j];

			// if the momentum term is greater than 0
			// the previous weighting needs to be taken into account
			if(mMomentum > 0)
			{
				if((int)mPrevOutWt.size() > prevIdx)
				{
					dWPrev = mPrevOutWt[prevIdx];
					
					// include a percentage of the previous weighting
					dW += mMomentum * dWPrev;
					
					// store the weighting 
					mPrevOutWt[prevIdx] = dW;
				}
				else
				{
					// store the first weighting
					mPrevOutWt.push_back(dW);
				}
			}

			// the total weight adjustment
			weights[j] += dW;
			prevIdx++;
		}

		wtConnect.setWeightVector(i, weights);
	}

	nNet.setWeightedConnect(wtConnect, n);
}

/////////////////////////////////////////////////////////////////////
// calcHiddenWtAdjust - calculates the weight adjustments for the 
//                      connections into the hidden layers
//
// hidErrSig (vector<vector<double> >) - the hidden unit errors
// inputVec (vector<double>) - the current training set input values
// nNet (NeuralNet) - the network undergoing training

void NNetTrainer::calcHiddenWtAdjust(const vector<vector<double> >& hidErrSig,
									 const vector<double>& inputVec, NeuralNet& nNet)
{
	vector<double> xVec;
	int maxHidLayIdx = nNet.getNumLayers() - 1, prevIdx = 0;

	// calculate the weight adjustments for the hidden layers
	for(int n = maxHidLayIdx; n >= 0; n--)
	{
		// get the weighted connections between the current layer and the previous hidden layer
		NNetWeightedConnect wtConnect;		
		nNet.getWeightedConnect(wtConnect, n);
		
		// get the hidden unit errors for the previous hidden layer
		// N.B. the hidden error signals are stored in reverse order
		vector<double> outErr = hidErrSig[maxHidLayIdx - n];

		if(n == 0)
		{
			// we are dealing with the input layer
			xVec = inputVec;
		}
		else
		{
			// we are dealing with a hidden layer
			nNet.getActivations(xVec, n - 1);
		}

		int nOut = wtConnect.getNumOutputNodes();

		// calculate the weight adjustments for each weighted connection output unit
		for(int i = 0; i < nOut; i++)
		{
			double ei = outErr[i];
			vector<double> weights;
			
			// get the output units weight vector
			wtConnect.getWeightVector(i, weights);

			// calculate the total weight adjustment
			for(int j = 0; j < (int)xVec.size(); j++)
			{
				double dWPrev = 0;

				// the weight adjustment calculation
				double dW = mLearnConst * ei * xVec[j];

				// if the momentum term is greater than 0
				// the previous weighting needs to be taken into account
				if(mMomentum > 0)
				{
					if((int)mPrevHidWt.size() > prevIdx)
					{
						dWPrev = mPrevHidWt[prevIdx];

						// include a percentage of the previous weighting
						dW += mMomentum * dWPrev;
						
						// store the weighting 
						mPrevHidWt[prevIdx] = dW;
					}
					else
					{
						// store the first weighting
						mPrevHidWt.push_back(dW);
					}
				}

				// the total weight adjustment
				weights[j] += dW;
				prevIdx++;
			}

			wtConnect.setWeightVector(i, weights);
		}

		nNet.setWeightedConnect(wtConnect, n);
	}
}

/////////////////////////////////////////////////////////////////////
// getGradient - gets the gradient of the activation function at the 
//               given value of x
//
// unitType (ActiveT) - the activation function type
// slope (double) - the activation function slope value
// amplify (double) - the activation function amplify value
// x (double) - calculates the gradient at this value
//
// returns the gradient of the activation function at the given value

double NNetTrainer::getGradient(ActiveT unitType, double slope, double amplify, double x)
{
	double gradient = 0;	
	double expMX, expMX1, tanMX, absMX1, grad;

	switch(unitType)
	{
	case kThreshold:    // 0 everywhere except the origin where the derivative is undefined!

		// return the value of the slope parameter if x = 0
		if(x == 0) gradient = slope;
		break;

	case kUnipolar:

		expMX = exp(-slope * x);
		expMX1 = 1 + expMX;

		gradient = (slope * expMX) / (expMX1 * expMX1);
		break;

	case kBipolar:

		expMX = exp(-slope * x);
		expMX1 = 1 + expMX;

		gradient = (2 * slope * expMX) / (expMX1 * expMX1);
		break;

	case kTanh:

		tanMX = tanh(slope * x);

		gradient = slope * (1 - (tanMX * tanMX));
		break;

	case kGauss:

		gradient = -2 * slope * x * exp(-slope * x * x);
		break;

	case kArctan:

		gradient = slope / (1 + slope * slope * x * x);
		break;

	case kSin:

		gradient = slope * cos(slope * x);
		break;

	case kCos:

		gradient = -slope * sin(slope * x);
		break;

	case kSinC:
		
		if(fabs(x) < 0.00001)
		{
			gradient = 0;
		}
		else
		{
			gradient = (slope * x * cos(slope * x) - sin(slope * x)) / (slope * x * x);
		}

		break;

	case kElliot:

		absMX1 = 1 + fabs(slope * x);

		gradient = (0.5 * slope) / (absMX1 * absMX1);
		break;

	case kLinear:

		gradient = slope;
		break;

	case kISRU:

		grad = 1 / sqrt(1 + slope * x * x);
		
		gradient = grad * grad * grad;
		break;

	case kSoftSign:

		absMX1 = 1 + fabs(slope * x);

		gradient = slope / (absMX1 * absMX1);
		break;

	case kSoftPlus:	

		expMX = exp(slope * x);

		gradient = (slope * expMX) / (1 + expMX);
		break;
	}

	return amplify * gradient;
}

/////////////////////////////////////////////////////////////////////
