/////////////////////////////////////////////////////////////////////
//
// Implements the NNetWeightedConnect class
//
// Author: Jason Jenkins
//
// This class is used by the neural network class (NeuralNet) and
// represents the weighted connections that link the layers of a 
// neural network together.
//
// The layers of a neural network are connected by a system of
// weighted connections. Each unit in a given layer of the network
// (excluding the output layer) has a single connection to every 
// unit in the next layer. These connections are initially given
// a random value which is then updated when the neural network
// is trained.
//
// In this class the weighted connections between two layers 
// consist of a given number of input and output nodes. If the
// first layer of a network contains three units and the second
// layer contains four to connect these layers together we require
// a weighted connection with three input nodes and four output
// nodes.
//
// Each input node is connected to every output node. The input 
// nodes have their values set by the setInputs method and these
// values represent the activated output of a particular layer
// within the network. The value of a specific output node is the
// result of applying the weighted connections between that output
// node and all the connected input nodes - each input node value is
// multiplied by the weighted connection between the input node and 
// the given output node in turn and the results are then summed to 
// give the output node value. The output node values can be obtained
// via the getOutputs method and these values can then be used by the
// network as the input values to the activation functions of the
// next layer.
//
// The weights of the connections between a given output node and
// the input nodes can be retrieved via the getWeightVector method
// and set via the setWeightVector method. These two methods are 
// typically called by the network training process.
//
/////////////////////////////////////////////////////////////////////

#include "NNetWeightedConnect.h"

/////////////////////////////////////////////////////////////////////
/// <summary>
/// default constructor
/// </summary>
/// 
NNetWeightedConnect::NNetWeightedConnect()
{
	// default connection settings
	mNumInNodes = -1;
	mNumOutNodes = -1;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// constructs a connection between the given number of nodes
/// </summary>
/// <param name="numInNodes">the number of input nodes</param>
/// <param name="numOutNodes">the number of output nodes</param>
/// 
NNetWeightedConnect::NNetWeightedConnect(int numInNodes, int numOutNodes)
{
	// ignore invalid data
	if(numInNodes > 0 && numOutNodes > 0)
	{
		mNumInNodes = numInNodes;
		mNumOutNodes = numOutNodes;

		initialiseWeights();
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// destructor
/// </summary>
///
NNetWeightedConnect::~NNetWeightedConnect()
{
}

/////////////////////////////////////////////////////////////////////
// Public Methods
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the number of input and output nodes - 
/// 
/// The weighted connections are randomly initialised over the range
/// -(initRange/2) to +(initRange/2).
/// </summary>
/// <param name="numInNodes">the number of input nodes</param>
/// <param name="numOutNodes">the number of output nodes</param>
/// <param name="initRange">the range used for random initialisation</param>
/// 
void NNetWeightedConnect::setNumNodes(int numInNodes, int numOutNodes, double initRange)
{
	// ignore invalid data
	if(numInNodes > 0 && numOutNodes > 0 && initRange > 0)
	{
		mNumInNodes = numInNodes;
		mNumOutNodes = numOutNodes;

		initialiseWeights(initRange);
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the input values for the weighted connection - 
/// 
/// The input value for the first input node is the first value stored
/// in the list and the input value for the second input node is the
/// second value stored in the list and so on.
/// </summary>
/// <param name="inputs">a vector of input values</param>
/// 
void NNetWeightedConnect::setInputs(const vector<double>& inputs)
{
	// make sure the size of the input vector corresponds to the number of input nodes
	if((int)inputs.size() == mNumInNodes)
	{
		if(mInputs.empty())
		{
			mInputs.resize(mNumInNodes);
		}

		// copy will overwrite the original values
		copy(inputs.begin(), inputs.end(), mInputs.begin());
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets the output values for the weighted connection - 
/// 
/// The output values are calculated by applying the weighted
/// connections to the input node values.
/// </summary>
/// <param name="outputs">the output values</param>
/// 
void NNetWeightedConnect::getOutputs(vector<double>& outputs)
{
	// resize the output vector if necessary
	if((int)outputs.size() != mNumOutNodes)
	{
		outputs.resize(mNumOutNodes);
	}

	calculateOutput();

	copy(mOutputs.begin(), mOutputs.end(), outputs.begin());
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets the weighted connections vector for a given output node - 
/// 
/// This method is typically called when training the network.
/// </summary>
/// <param name="node">the index of the output node</param>
/// <param name="weights">the weighted connections vector</param>
/// 
void NNetWeightedConnect::getWeightVector(int node, vector<double>& weights)
{
	if(node < (int)mWeights.size() && node >= 0)
	{
		vector<double> vecWeights = mWeights[node];

		if(weights.size() != vecWeights.size())
		{
			weights.resize(vecWeights.size());
		}

		copy(vecWeights.begin(), vecWeights.end(), weights.begin());
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the weighted connections vector for a given output node - 
/// 
/// This method is typically called by the training process to update the weighted connections.
/// </summary>
/// <param name="node">the index of the output node</param>
/// <param name="weights">the weighted connections vector</param>
/// 
void NNetWeightedConnect::setWeightVector(int node, const vector<double>& weights)
{
	if(node < (int)mWeights.size() && node >= 0)
	{
		if(mWeights[node].size() == weights.size())
		{
			for(int i = 0; i < (int)weights.size(); i++)
			{
				mWeights[node][i] = weights[i];
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////
// Private Methods
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/// <summary>
/// randomly initialises the weighted connections - 
/// 
/// The weighted connections are randomly initialised over the range
/// -(initRange/2) to +(initRange/2).
/// </summary>
/// <param name="initRange">the range used for random initialisation</param>
/// 
void NNetWeightedConnect::initialiseWeights(double initRange)
{	
	// initialise a weight vector for each of the output nodes
	for(int i = 0; i < mNumOutNodes; i++)
	{		
		vector<double> initVec;

		// the size of the vector is equal to the number of input nodes
		for(int j = 0; j < mNumInNodes; j++)
		{
			double initVal = rand();

			// randomly iniialise a vector component
			initVal = initRange * (initVal / RAND_MAX) - (initRange / 2);

			initVec.push_back(initVal);
		}

		mWeights.push_back(initVec);
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// calculates the output values for all the output nodes
/// </summary>
/// 
void NNetWeightedConnect::calculateOutput()
{
	if(!mOutputs.empty())
	{
		mOutputs.erase(mOutputs.begin(), mOutputs.end());
	}

	for(int i = 0; i < mNumOutNodes; i++)
	{
		double component = getNodeValue(i);

		mOutputs.push_back(component);
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// calculates the output value for the given output node
/// </summary>
/// <param name="node">the index of the output node</param>
///
/// <returns>the value of the output node</returns>
/// 
double NNetWeightedConnect::getNodeValue(int node)
{
	double value = 0;
	vector<double> nodeVec(mWeights[node].size());

	copy(mWeights[node].begin(), mWeights[node].end(), nodeVec.begin());

	if(mNumInNodes == (int)nodeVec.size())
	{
		for(int i = 0; i < mNumInNodes; i++)
		{
			value += nodeVec[i] * mInputs[i];
		}
	}

	return value;
}

/////////////////////////////////////////////////////////////////////
