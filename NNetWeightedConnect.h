/////////////////////////////////////////////////////////////////////
//
// Defines the NNetWeightedConnect class
//
// Author: Jason Jenkins
//
// This class is used by the neural network class (NeuralNet) and
// represents the weighted connections that link the layers of a 
// neural network together.
//
/////////////////////////////////////////////////////////////////////

#pragma once

/////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>

/////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////

class NNetWeightedConnect
{
public:
	NNetWeightedConnect();	
	virtual ~NNetWeightedConnect();

	// constructs a connection between the given number of nodes
	NNetWeightedConnect(int numInNodes, int numOutNodes);	

	// sets the number of input and output nodes
	void setNumNodes(int numInNodes, int numOutNodes, double initRange = 2.0);

	// returns the number of input nodes
	int getNumInputNodes () const { return mNumInNodes; }

	// returns the number of output nodes
	int getNumOutputNodes () const { return mNumOutNodes; }

	// sets the input values for the weighted connection
	void setInputs(const vector<double>& inputs);

	// gets the output values for the weighted connection
	void getOutputs(vector<double>& outputs);

	// gets the weighted connections vector for a given output node 
	void getWeightVector(int node, vector<double>& weights);

	// sets the weighted connections vector for a given output node 
	void setWeightVector(int node, const vector<double>& weights);

private:
	// randomly initialises the weighted connections
	void initialiseWeights(double initRange = 2.0);
	
	// calculates the output values for all the output nodes
	void calculateOutput();
	
	// calculates the output value for the given output node
	double getNodeValue(int node);

private:
	int mNumInNodes;    // the number of input nodes
	int mNumOutNodes;   // the number of output bodes

	vector<double> mInputs;     // the input values
	vector<double> mOutputs;    // the output values

	// the weighted connection values 
	vector<vector<double> > mWeights;	
};

/////////////////////////////////////////////////////////////////////
