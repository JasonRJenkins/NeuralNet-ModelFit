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
/// <summary>
/// This class is used by the neural network class (NeuralNet) and represents
/// the weighted connections that link the layers of a neural network together.
/// </summary>
/// 
class NNetWeightedConnect
{
public:
	NNetWeightedConnect();	
	virtual ~NNetWeightedConnect();

	// constructs a connection between the given number of nodes
	NNetWeightedConnect(int numInNodes, int numOutNodes);	

	// sets the number of input and output nodes
	void setNumNodes(int numInNodes, int numOutNodes, double initRange = 2.0);

	/// <summary>
	/// <returns>the number of input nodes</returns>
	/// </summary>
	int getNumInputNodes () const { return mNumInNodes; }

	/// <summary>
	/// <returns>the number of output nodes</returns>
	/// </summary>
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
	/// <summary>the number of input nodes</summary>
	int mNumInNodes;

	/// <summary>the number of output nodes</summary>
	int mNumOutNodes;

	/// <summary>the input values</summary>
	vector<double> mInputs;

	/// <summary>the output values</summary>
	vector<double> mOutputs;

	/// <summary>the weighted connection values</summary>
	vector<vector<double> > mWeights;	
};

/////////////////////////////////////////////////////////////////////
