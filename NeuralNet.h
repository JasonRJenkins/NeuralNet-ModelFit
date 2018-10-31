/////////////////////////////////////////////////////////////////////
//
// Defines the NeuralNet class
//
// Author: Jason Jenkins
//
// This class is a representation of a feedforward neural network.
//
/////////////////////////////////////////////////////////////////////

#pragma once

/////////////////////////////////////////////////////////////////////

#include <string>

/////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////

#include "NNetUnit.h"
#include "NNetWeightedConnect.h"

/////////////////////////////////////////////////////////////////////

class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();

	// constructs a NeuralNet object from a file
	NeuralNet(const string& fName);
	
	// sets the number of input units
	void setNumInputs(int numInputs);

	// sets the number of output units
	void setNumOutputs(int numOutputs);

	// set the output layer unit activation function type
	void setOutputUnitType(ActiveT unitType);

	// set the output layer unit activation function slope value
	void setOutputUnitSlope(double slope);

	// set the output layer unit activation function amplify value
	void setOutputUnitAmplify(double amplify);
	
	// returns the number of input units
	int getNumInputs() const { return mNumInputs; }
	
	// returns the number of output units
	int getNumOutputs() const { return mNumOutputs; }
	
	// returns the number of hidden layers
	int getNumLayers() const { return mNumLayers; }

	// returns the output layer unit activation function type
	ActiveT getOutputUnitType() const { return mOutUnitType; }

	// returns the output layer unit activation function slope value
	double getOutputUnitSlope() const { return mOutUnitSlope; };

	// returns the output layer unit activation function amplify value
	double getOutputUnitAmplify() const { return mOutUnitAmplify; };

	// adds a new hidden layer
	int addLayer(int numUnits, ActiveT unitType = kUnipolar, 
				 double initRange = 2.0, double slope = 1.0, double amplify = 1.0);
	
	// gets the details of the specified hidden layer
	void getLayerDetails(int n, ActiveT& unitType, double& slope, double& amplify);

	// gets the response of the network to the given input	
	void getResponse(const vector<double>& inputs, vector<double>& outputs);
	
	// gets the activation values for a specified layer
	void getActivations(vector<double>& activations, int layer);

	// gets the unit input values for a specified layer
	void getUnitInputs(vector<double>& inputs, int layer);

	// gets the weighted connections for a specified layer
	void getWeightedConnect(NNetWeightedConnect& wtConnect, int layer);

	// sets the weighted connections for a specified layer
	void setWeightedConnect(const NNetWeightedConnect& wtConnect, int layer);	
	
	// serializes the network and writes it to a file
	int writeToFile(const string& fname);

private:
	// generates a string representation of the network
	string serialize();

	// instantiates a network from a string representation
	void deserialize(const string& inData);
	
private:
	int mNumInputs;    // the number of input units
	int mNumOutputs;   // the number of output units
	int mNumLayers;    // the number of hidden layers
	
	// the output layer unit activation function type
	ActiveT mOutUnitType;
	
	// the output layer unit activation function slope value
	double mOutUnitSlope;

	// the output layer unit activation function amplify value
	double mOutUnitAmplify;

	// the weighted connections linking the network layers
	vector<NNetWeightedConnect> mLayers;

	// the activation values for each of the network layers
	vector<vector<double> > mActivations;

	// the input values for the layer activation functions
	vector<vector<double> > mUnitInputs;

	// the hidden layer unit activation function types
	vector<ActiveT> mActiveUnits;
	
	// the hidden layer unit activation function slope values
	vector<double> mActiveSlope;
	
	// the hidden layer unit activation function amplify values
	vector<double> mActiveAmplify;
};

/////////////////////////////////////////////////////////////////////
