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
// the available activation functions as an enumerated type

typedef enum { kThreshold, kUnipolar, kBipolar, kTanh, kGauss, kArctan, kSin,
			   kCos, kSinC, kElliot, kLinear, kISRU, kSoftSign, kSoftPlus } ActiveT;

/////////////////////////////////////////////////////////////////////

class NNetUnit
{
public:
	NNetUnit();
	virtual ~NNetUnit();

	// constructs a neuron with the given activation function and settings
	NNetUnit(ActiveT activationMode, double slope = 1.0, double amplify = 1.0);
	
	// sets the activation function type
	void setActivationType(ActiveT activationType) { mActivationType = activationType; };

	// sets the input value of the neuron
	void setInput(double input) { mInput = input; }

	// sets the slope parameter of the activation function
	void setSlope(double slope);

	// sets the amplify parameter of the activation function
	void setAmplify(double amplify);

	// returns activation value of the neuron
	double getActivation();

private:
	ActiveT mActivationType;    // the unit activation function type
	double mInput;              // the unit input value
	double mSlope;              // the activation function slope setting
	double mAmplify;            // the activation function amplify setting
};

/////////////////////////////////////////////////////////////////////
