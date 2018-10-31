/////////////////////////////////////////////////////////////////////
//
// Defines the NNetTrainer class
//
// Author: Jason Jenkins
//
// This class provides a framework for training a neural network.
//
/////////////////////////////////////////////////////////////////////

#pragma once

/////////////////////////////////////////////////////////////////////

#include "NeuralNet.h"

/////////////////////////////////////////////////////////////////////

class NNetTrainer
{
public:
	NNetTrainer();
	virtual ~NNetTrainer();

	// sets the learning constant training parameter
	void setLearningConstant(double learnConst);

	// returns the learning constant training parameter
	double getLearningConst() const { return mLearnConst; }

	// sets the momentum training parameter
	void setMomentum(double momentum);

	// returns the momentum training parameter
	double getMomentum() const { return mMomentum; }

	// returns the total network error
	double getNetError() const { return mNetError; }

	// resets the total network error to zero
	void resetNetError() { mNetError = 0; }

	// trains the supplied neural network
	void trainNeuralNet(NeuralNet& nNet);

	// adds an individual training vector and 
	// corresponding target vector to the training set
	void addToTrainingSet(const vector<double>& inVec, 
						  const vector<double>& outVec);

	// adds a complete set of training vectors and
	// corresponding target vectors to the trainer
	void addNewTrainingSet(const vector<vector<double> >& inVecs, 
						   const vector<vector<double> >& outVecs);

private:
	// calculates the network error between a given vector of 
	// response values and the corresponding vector of target values
	double calcNetworkError(const vector<double>& response, int nTarget);

	// calculates the error signal on each individual output unit in the output layer
	void calcOutputError(NeuralNet& nNet, vector<double>& outErr,
						 const vector<double>& response, int nTarget);
	
	// calculates the error signal on each individual unit within the networks hidden layers
	void calcHiddenError(vector<vector<double> >& hidErr, 
						 const vector<double>& outErr, NeuralNet& nNet);

	// calculates the weight adjustments for the connections into the output layer
	void calcOutputWtAdjust(const vector<double>& outErr, NeuralNet& nNet);

	// calculates the weight adjustments for the connections into the hidden layers
	void calcHiddenWtAdjust(const vector<vector<double> >& hidErrSig, 
							const vector<double>& inputVec, NeuralNet& nNet);

	// returns the gradient of the activation function at the given value
	double getGradient(ActiveT unitType, double slope, double amplify, double x);

private:
	double mNetError;		// the network error
	double mLearnConst;		// the learning constant
	double mMomentum;		// the momentum parameter

	// keeps track of the output layer weightings for use by the momentum term
	vector<double> mPrevOutWt;

	// keeps track of the hidden layers weightings for use by the momentum term
	vector<double> mPrevHidWt;

	// the training set
	vector<vector<double> > mTrainInput;    // the training set input values
	vector<vector<double> > mTrainTarget;   // the training set target values
};

/////////////////////////////////////////////////////////////////////
