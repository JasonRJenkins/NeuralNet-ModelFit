/////////////////////////////////////////////////////////////////////
//
// Implements the NeuralNet class
//
// Author: Jason Jenkins
//
// This class is a representation of a feedforward neural network.
//
// This class enables a neural network to be built comprising single 
// or multiple input and output values along with one or more hidden
// layers.
//
// The output and hidden layers can consist of any number of units
// or neurons and each layer can be given their own activation 
// function, to be used by all the units in that layer, from a 
// selection of available types:
//
// Threshold, Unipolar, Bipolar, Tanh, Gaussian, Arctan, Sine,
// Cosine, Sinc, Elliot, Linear, ISRU, SoftSign and SoftPlus.
//
// The activation function values can also be modified using two
// parameters: slope and amplify (see NNetUnit.cpp for details). 
//
// A NeuralNet object can be serialized to and de-serialized from
// a string representation which can be written to or read from a
// file. This allows a neural network to be used once training is
// complete or to continue training if required.
//
// The following code creates a neural network with 2 input units, 3
// output units and 2 hidden layers with 4 and 6 units respectively.
// The output units will use unipolar activation functions and the
// the hidden layer units will both use bipolar activation functions.
/*		
		NeuralNet net;		
		net.setNumInputs(2);
		net.setNumOutputs(3);
		net.setOutputUnitType(kUnipolar);
		net.addLayer(4, kBipolar);		// the first hidden layer
		net.addLayer(6, kBipolar);		// the second hidden layer
*/
// To use the neural network, once it has been trained, populate a
// vector with the desired input values and call the getResponse
// method to populate another vector with the output values.
/*
		vector<double> inputs;
		vector<double> outputs;
		inputs.push_back(0.5);
		inputs.push_back(0.2);
		net.getResponse(inputs, outputs);

		double outputValue1 = outputs[0];
		double outputValue2 = outputs[1];
		double outputValue3 = outputs[2];
*/
//
/////////////////////////////////////////////////////////////////////

#include "NeuralNet.h"

/////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

/////////////////////////////////////////////////////////////////////
/// <summary>
/// default constructor
/// </summary>
/// 
NeuralNet::NeuralNet()
{
	mNumInputs = 0;
	mNumOutputs = 0;
	mNumLayers = 0;

	// default output unit settings
	mOutUnitType = kThreshold;
	mOutUnitSlope = 1.0;
	mOutUnitAmplify = 1.0;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// constructs a NeuralNet object from a file containing a network in 
/// serialized form
/// </summary>
/// <param name="fname">the file containing the serialized data</param>
/// 
NeuralNet::NeuralNet(const string& fname)
{
	ifstream inFile(fname);

	if(inFile.good())
	{		
		// find the length of the file
		inFile.seekg (0, inFile.end);
		int length = (int)inFile.tellg();
		inFile.seekg (0, inFile.beg);

		// create a suitably sized buffer
		char* buffer = new char[length];

		// read the data into the buffer
		inFile.read(buffer, length);

		// instantiate the network
		deserialize(buffer);

		delete[] buffer;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// destructor
/// </summary>
///
NeuralNet::~NeuralNet()
{
}

/////////////////////////////////////////////////////////////////////
// Public Methods
/////////////////////////////////////////////////////////////////////

/// <summary>
/// clears a NeuralNet object ready for re-use
/// </summary>
/// 
void NeuralNet::clearNeuralNetwork()
{
	mNumInputs = 0;
	mNumOutputs = 0;
	mNumLayers = 0;
	mOutUnitType = kThreshold;
	mOutUnitSlope = 1;
	mOutUnitAmplify = 1;

	mLayers.clear();
	mActivations.clear();
	mUnitInputs.clear();
	mActiveUnits.clear();
	mActiveSlope.clear();
	mActiveAmplify.clear();
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the number of input units
/// </summary>
/// <param name="numInputs">the number of input units</param>
///
void NeuralNet::setNumInputs(int numInputs)
{
	// ignore invalid values
	if(numInputs > 0)
	{
		mNumInputs = numInputs;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the number of output units
/// </summary>
/// <param name="numOutputs">the number of output units</param>
///
void NeuralNet::setNumOutputs(int numOutputs)
{
	// ignore invalid values
	if(numOutputs > 0)
	{
		mNumOutputs = numOutputs;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the output layer units activation function type
/// </summary>
/// <param name="unitType">the activation function type</param>
///
void NeuralNet::setOutputUnitType(ActiveT unitType)
{
	mOutUnitType = unitType;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the output layer units activation function slope value
/// </summary>
/// <param name="slope">the slope value for the activation function</param>
///
void NeuralNet::setOutputUnitSlope(double slope)
{
	// ignore invalid values
	if(slope > 0)
	{
		mOutUnitSlope = slope;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the output layer units activation function amplify value
/// </summary>
/// <param name="amplify">the amplify value for the activation function</param>
///
void NeuralNet::setOutputUnitAmplify(double amplify)
{
	// ignore invalid values
	if(amplify > 0)
	{
		mOutUnitAmplify = amplify;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// adds a new hidden layer - 
/// 
/// The hidden layers are stored in the order of calls to this method
/// so the first call to addlayer creates the first hidden layer, the
/// second call creates the second layer and so on.
/// </summary>
/// <param name="numUnits">the number of units in the hidden layer</param>
/// <param name="unitType">the layer unit activation function type 
///                        (defaults to unipolar)</param>
/// <param name="initRange">the range of the initial weighted connection
///                         values (defaults to 2 coressponding to -1 to +1)</param>
/// <param name="slope">the layer unit activation function slope value 
///                     (defaults to 1.0)</param>
/// <param name="amplify">the layer unit activation function amplify 
///                       value (defaults to 1.0)</param>
///
/// <returns>0 if the layer is successfully added otherwise -1</returns>
/// 
int NeuralNet::addLayer(int numUnits, ActiveT unitType, double initRange, double slope, double amplify)
{
	NNetWeightedConnect connect, output;

	// ignore invalid values
	if(numUnits > 0 && initRange > 0 && slope > 0 && amplify > 0)
	{
		if(mNumLayers == 0)
		{
			// configure the first hidden layer
			if(mNumInputs > 0)
			{
				// set up the weighted connections between the input and the first layer
				// the weighted connections are initialised with random values in the
				// range: -(initRange / 2) to +(initRange / 2)
				connect.setNumNodes(mNumInputs, numUnits, initRange);

				// store the unit type for the layer
				mActiveUnits.push_back(unitType);

				// store the steepness of the activation function's slope
				mActiveSlope.push_back(slope);

				// store the amplification factor of the activation function
				mActiveAmplify.push_back(amplify);

				mNumLayers++;			
			}
			else
			{
				return -1;
			}
		}
		else
		{		
			// configure subsequent hidden layers
			if(mNumLayers > 0)
			{
				int nInputs = mLayers[mNumLayers - 1].getNumOutputNodes();

				// set up the weighted connections between the previous layer and the new one
				// the weighted connections are initialised with random values in the
				// range: -(initRange / 2) to +(initRange / 2)
				connect.setNumNodes(nInputs, numUnits, initRange);

				// store the unit type for the layer
				mActiveUnits.push_back(unitType);

				// store the steepness of the activation function's slope
				mActiveSlope.push_back(slope);

				// store the amplification factor of the activation function
				mActiveAmplify.push_back(amplify);

				mNumLayers++;
			}
			else
			{
				return -1;
			}
		}

		// connect the last hidden layer to the output layer
		if(mNumLayers > 1)
		{
			// overwrite the old output connections
			mLayers[mNumLayers - 1] = connect;
		}
		else
		{
			// add the connections for the first layer
			mLayers.push_back(connect);
		}

		// set up the weighted connections between the last layer and the output
		output.setNumNodes(numUnits, mNumOutputs, initRange);

		// add the output connections
		mLayers.push_back(output);
	}
	else
	{
		return -1;
	}

	return 0;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets the details of the specified hidden layer
/// </summary>
/// <param name="n">the specified hidden layer index</param>
/// <param name="unitType">the layer unit activation function type</param>
/// <param name="slope">the layer unit activation function slope value</param>
/// <param name="amplify">the layer unit activation function amplify value</param>
/// 
void NeuralNet::getLayerDetails(int n, ActiveT& unitType, double& slope, double& amplify)
{
	if(n >= 0 && n < mNumLayers)
	{
		unitType = mActiveUnits[n];
		slope = mActiveSlope[n];
		amplify = mActiveAmplify[n];
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets the response of the network to the given input - 
/// 
/// The number of elements in the inputs vector should correspond to 
/// the number of the input units.  If the inputs vector contains 
/// more elements than this, the additional input values are ignored.
/// </summary>
/// <param name="inputs">the network input values</param>
/// <param name="outputs">the network output values</param>
/// 
void NeuralNet::getResponse(const vector<double>& inputs, vector<double>& outputs)
{
	vector<double> inputVec;
	vector<double> outputVec;
	NNetWeightedConnect connect;

	if((int)inputs.size() >= mNumInputs && mNumLayers > 0)
	{
		// clear any old activation and unit input values
		mActivations.clear();
		mUnitInputs.clear();

		// 'load' the input vector 
		for(int i = 0; i < mNumInputs; i++)
		{
			inputVec.push_back(inputs[i]);
		}

		// get the weighted connections between the input layer and first layer
		connect = mLayers[0];
		
		// apply the weighted connections
		connect.setInputs(inputVec);
		connect.getOutputs(outputVec);

		// store the output vector - this contains the unit input values
		mUnitInputs.push_back(outputVec);

		// clear the input vector so it can be used to hold the input for the next layer
		inputVec.clear();

		// set the unit type, slope and amplification for the first layer
		NNetUnit unit(mActiveUnits[0], mActiveSlope[0], mActiveAmplify[0]);

		// activate the net units
		for(int i = 0; i < (int)outputVec.size(); i++)
		{
			unit.setInput(outputVec[i]);
			inputVec.push_back(unit.getActivation());
		}

		// store the activations
		mActivations.push_back(inputVec);

		// propagate the data through the remaining layers
		for(int i = 1; i <= mNumLayers; i++)	// use <= to include the output layer
		{			
			// get the weighted connections linking the next layer
			connect = mLayers[i];			

			// apply the weighted connections
			outputVec.clear();
			connect.setInputs(inputVec);
			connect.getOutputs(outputVec);
			inputVec.clear();

			// store the output vector - this contains the unit input values
			mUnitInputs.push_back(outputVec);

			if(i < mNumLayers)
			{
				// set the unit type, slope and amplification for the next hidden layer
				unit.setActivationType(mActiveUnits[i]);
				unit.setSlope(mActiveSlope[i]);
				unit.setAmplify(mActiveAmplify[i]);
			}
			else
			{
				// set the unit type, slope and amplification for the output layer
				unit.setActivationType(mOutUnitType);
				unit.setSlope(mOutUnitSlope);
				unit.setAmplify(mOutUnitAmplify);
			}

			// activate the net units
			for(int j = 0; j < (int)outputVec.size(); j++)
			{
				unit.setInput(outputVec[j]);
				inputVec.push_back(unit.getActivation());
			}

			// store the activations
			mActivations.push_back(inputVec);
		}
	
		// copy the results into the output vector
		if(outputs.size() > 0) outputs.clear();
		outputs = inputVec;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets the activation values for a specified layer - 
/// 
/// This method is typically called by the training process to access
/// the activation values of the hidden and output layers.
/// </summary>
/// <param name="activations">the activation values for the layer</param>
/// <param name="layer">the specified layer</param>
/// 
void NeuralNet::getActivations(vector<double>& activations, int layer)
{
	if(layer >= 0 && layer < (int)mActivations.size())
	{
		activations = mActivations[layer];
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets the unit input values for a specified layer - 
/// 
/// This method is typically called by the training process to access
/// the input values to the hidden and output layer activation functions.
/// </summary>
/// <param name="inputs">the unit input values for the layer</param>
/// <param name="layer">the specified layer</param>
/// 
void NeuralNet::getUnitInputs(vector<double>& inputs, int layer)
{
	if(layer >= 0 && layer < (int)mUnitInputs.size())
	{
		inputs = mUnitInputs[layer];
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets the weighted connections for a specified layer - 
/// 
/// This method is typically called by the training process to access 
/// the weighted connections.
/// </summary>
/// <param name="wtConnect">the weighted connections between the 
///                         specified layer and the next sequential
///                         layer in the network.</param>
/// <param name="layer">the specified layer</param>
/// 
void NeuralNet::getWeightedConnect(NNetWeightedConnect& wtConnect, int layer)
{
	if(layer >= 0 && layer < (int)mLayers.size())
	{
		wtConnect = mLayers[layer];
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the weighted connections for a specified layer - 
/// 
/// This method is typically called by the training process to update 
/// the weighted connections.
/// </summary>
/// <param name="wtConnect">the weighted connections between the 
///                         specified layer and the next sequential 
///                         layer in the network</param>
/// <param name="layer">the specified layer</param>
/// 
void NeuralNet::setWeightedConnect(const NNetWeightedConnect& wtConnect, int layer)
{
	if(layer >= 0 && layer < (int)mLayers.size())
	{
		mLayers[layer] = wtConnect;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// serializes this network and writes it to a file
/// </summary>
/// <param name="fname">the file to write the data to</param>
/// <returns>0 if successful otherwise -1</returns>
/// 
int NeuralNet::writeToFile(const string& fname)
{
	ofstream outFile(fname);

	if(outFile.good())
	{
		outFile << serialize();
	}
	else
	{
		return -1;
	}

	return 0;
}

/////////////////////////////////////////////////////////////////////
// Private Methods
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/// <summary>
/// generates a string representation of this network
/// </summary>
/// <returns>a string representation of this network</returns>
/// 
string NeuralNet::serialize()
{
	vector<double> weights;
	ostringstream outStream;		

	outStream << std::setprecision(16);

	// serialize the main details
	outStream << mNumInputs << " " <<
			     mNumOutputs << " " <<
				 mNumLayers << " " <<
				 mOutUnitType << " " <<
				 mOutUnitSlope << " " <<
				 mOutUnitAmplify << " ";

	// serialize the layer data
	for(int i = 0; i <= mNumLayers; i++)		// use <= to include the output layer
	{
		NNetWeightedConnect connect = mLayers[i];
		int nIn = connect.getNumInputNodes();
		int nOut = connect.getNumOutputNodes();
		int nUnit = 0;
		double sUnit = 0.0, aUnit = 0.0;

		// get the unit type, slope and amplification for the hidden layer
		if(i < mNumLayers) nUnit = (int)mActiveUnits[i];
		if(i < mNumLayers) sUnit = mActiveSlope[i];
		if(i < mNumLayers) aUnit = mActiveAmplify[i];

		outStream << "L " << nIn << " " << nOut << " " << nUnit << " " << sUnit << " " << aUnit << " ";

		for(int j = 0; j < nOut; j++)
		{
			connect.getWeightVector(j, weights);

			for(int k = 0; k < nIn; k++)
			{
				outStream << weights[k] << " ";
			}
		}
	}

	// terminate the output string
	outStream << endl;

	return outStream.str();
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// instantiates this network from a given string representation
/// </summary>
/// <param name="inData">the given string representation of the network</param>
/// 
void NeuralNet::deserialize(const string& inData)
{	
	istringstream inStream(inData);	

	if(!inStream.good())
	{
		cerr << "Error deserializing!" << endl;
	}
	else
	{
		int outUnitType;

		// deserialize the main details
		inStream >> mNumInputs;
		inStream >> mNumOutputs;
		inStream >> mNumLayers;
		inStream >> outUnitType;
		inStream >> mOutUnitSlope;
		inStream >> mOutUnitAmplify;

		mOutUnitType = (ActiveT)outUnitType;

		// deserialize the layer data
		for(int i = 0; i <= mNumLayers; i++)		// use <= to include the output layer
		{
			char delim;
			int nIn, nOut, nUnit;
			double sUnit, aUnit;
			
			inStream >> delim;
 			inStream >> nIn;
			inStream >> nOut;
			inStream >> nUnit;
			inStream >> sUnit;
			inStream >> aUnit;

			NNetWeightedConnect connect(nIn, nOut);

			for(int j = 0; j < nOut; j++)
			{
				vector<double> weights;

				for(int k = 0; k < nIn; k++)
				{
					double wgt;

					inStream >> wgt;

					weights.push_back(wgt);
				}

				connect.setWeightVector(j, weights);
			}

			mLayers.push_back(connect);
			mActiveUnits.push_back((ActiveT)nUnit);
			mActiveSlope.push_back(sUnit);
			mActiveAmplify.push_back(aUnit);
		}
	}
}

/////////////////////////////////////////////////////////////////////
