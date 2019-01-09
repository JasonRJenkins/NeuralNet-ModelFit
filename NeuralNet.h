/////////////////////////////////////////////////////////////////////
//
// Defines the NeuralNet class
//
// Author: Jason Jenkins
//
// This class is a representation of a feed forward neural network.
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
/// <summary>
/// This class is a representation of a feed forward neural network.
/// </summary>
/// 
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
	
	/// <summary>
	/// <returns>the number of input units</returns>
	/// </summary>
	int getNumInputs() const { return mNumInputs; }
	
	/// <summary>
	/// <returns>the number of output units</returns>
	/// </summary>
	int getNumOutputs() const { return mNumOutputs; }
	
	/// <summary>
	/// <returns>the number of hidden layers</returns>
	/// </summary>
	int getNumLayers() const { return mNumLayers; }

	/// <summary>
	/// <returns>the output layer unit activation function type</returns>
	/// </summary>
	ActiveT getOutputUnitType() const { return mOutUnitType; }

	/// <summary>
	/// <returns>the output layer unit activation function slope value</returns>
	/// </summary>
	double getOutputUnitSlope() const { return mOutUnitSlope; };

	/// <summary>
	/// <returns>the output layer unit activation function amplify value</returns>
	/// </summary>
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
	/// <summary>the number of input units</summary>
	int mNumInputs;

	/// <summary>the number of output units</summary>
	int mNumOutputs;

	/// <summary>the number of hidden layers</summary>
	int mNumLayers;
	
	/// <summary>the output layer unit activation function type</summary>
	ActiveT mOutUnitType;
	
	/// <summary>the output layer unit activation function slope value</summary>
	double mOutUnitSlope;

	/// <summary>the output layer unit activation function amplify value</summary>
	double mOutUnitAmplify;

	/// <summary>the weighted connections linking the network layers</summary>
	vector<NNetWeightedConnect> mLayers;

	/// <summary>the activation values for each of the network layers</summary>
	vector<vector<double> > mActivations;

	/// <summary>the input values for the layer activation functions</summary>
	vector<vector<double> > mUnitInputs;

	/// <summary>the hidden layer unit activation function types</summary>
	vector<ActiveT> mActiveUnits;
	
	/// <summary>the hidden layer unit activation function slope values</summary>
	vector<double> mActiveSlope;
	
	/// <summary>the hidden layer unit activation function amplify values</summary>
	vector<double> mActiveAmplify;
};

/////////////////////////////////////////////////////////////////////
