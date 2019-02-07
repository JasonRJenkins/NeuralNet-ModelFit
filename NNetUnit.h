/////////////////////////////////////////////////////////////////////
//
// Defines the NNetUnit class
//
// Author: Jason Jenkins
//
// This class is used by the neural network class (NeuralNet) and
// represents the basic neural network unit or neuron.
//
/////////////////////////////////////////////////////////////////////

#pragma once

/////////////////////////////////////////////////////////////////////

#include <string>

/////////////////////////////////////////////////////////////////////
/// The available activation functions as an enumerated type

typedef enum { kThreshold, kUnipolar, kBipolar, kTanh, kGauss, kArctan, kSin,
			   kCos, kSinC, kElliot, kLinear, kISRU, kSoftSign, kSoftPlus } ActiveT;

/////////////////////////////////////////////////////////////////////
/// <summary>
/// This class is used by the neural network class (NeuralNet) and
/// represents the basic neural network unit or neuron.
/// </summary>
/// 
class NNetUnit
{
public:
	NNetUnit();
	virtual ~NNetUnit();

	// constructs a neuron with the given activation function and settings
	NNetUnit(ActiveT activationMode, double slope = 1.0, double amplify = 1.0);
	
	/// <summary>sets the activation function type</summary>
	void setActivationType(ActiveT activationType) { mActivationType = activationType; };

	/// <summary>sets the input value of the neuron</summary>
	void setInput(double input) { mInput = input; }

	// sets the slope parameter of the activation function
	void setSlope(double slope);

	// sets the amplify parameter of the activation function
	void setAmplify(double amplify);

	// returns activation value of the neuron
	double getActivation();

	// converts an ActiveT enumeration to its string representation
	static std::string ActiveTtoString(const ActiveT activEnum);

private:
	/// <summary>the unit activation function type</summary>
	ActiveT mActivationType;

	/// <summary>the unit input value</summary>
	double mInput;

	/// <summary>the activation function slope setting</summary>
	double mSlope;

	/// <summary>the activation function amplify setting</summary>
	double mAmplify;
};

/////////////////////////////////////////////////////////////////////
