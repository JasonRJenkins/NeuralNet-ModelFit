/////////////////////////////////////////////////////////////////////
//
// Implements the NNetUnit class.
//
// Author: Jason Jenkins
//
// This class is used by the neural network class (NeuralNet) and
// represents the basic neural network unit or neuron.
//
// A unit or neuron can be assigned one of a number activation 
// functions from a selection of available types:
//
// Threshold, Unipolar, Bipolar, Tanh, Gaussian, Arctan, Sine,
// Cosine, Sinc, Elliot, Linear, ISRU, SoftSign and SoftPlus.
//
// The output of the activation function can also be modified using 
// two parameters: slope and amplify.
//
// The amplify parameter can be used to increase or decrease the
// activation value which alters the range of the function.
//
// The slope parameter, sometimes termed lamda, can be used to 
// adjust the sensitivity of the activation function and its 
// effect depends on the particular function - For the sigmoidal 
// functions: unipolar, bipolar, tanh, arctan, elliot and soft 
// sign increasing the value of the slope parameter will increase
// the slope of the curve at the origin:-
//
//           decreasing <--- slope ---> increasing
//               ---                       ---
//              /                          |
//             /                           | 
//          ---                          ---
//
// For the sigmoidal function ISRU (inverse square root unit)
// changing the value of the slope parameter has the opposite 
// effect on the slope of the curve at the origin. Changing  
// the slope also effects the range of the activation value:
// default range: -1 / sqrt(mSlope) to 1 / sqrt(mSlope)
//
// For the periodic functions: sine, cosine and sinc increasing the 
// slope value will decrease the spatial period of the function i.e.
// the distance along the x-axis between repeated values is reduced.
// Decreasing the slope value has the opposite effect.
//
// For the Gaussian function - with a symmetric 'bell curve' shape -
// the slope value effects the width of 'bell': increasing the slope
// parameter will decrease the width of the 'bell' and vice versa.
//
// The effect of the slope parameter on the linear and soft plus
// functions is to increase or decrease the steepness or gradient
// of the functions as the slope parameter increases or decreases.
//
// For the threshold function the slope parameter has a similar
// effect to the amplify parameter. The threshold function usually
// returns one of two values - 0 or 1 - which can be adjusted by
// using the slope (or amplify) parameter to return 0 or the value
// of the product of the slope and amplify parameters.
//
// In all cases except ISRU and threshold varying the slope 
// parameter does not affect the default range of the particular 
// activation function - to alter the range of these functions you 
// need to vary the amplify parameter.
//
/////////////////////////////////////////////////////////////////////

#include <math.h>

/////////////////////////////////////////////////////////////////////

#include "NNetUnit.h"

/////////////////////////////////////////////////////////////////////
// default constructor

NNetUnit::NNetUnit()
{
	// default unit settings
	mInput = -1;
	mActivationType = kThreshold;
	mSlope = 1.0;
	mAmplify = 1.0;
}

/////////////////////////////////////////////////////////////////////
// constructs a neuron with the given activation function and settings
//
// activationMode (ActiveT) - the unit activation function type
// slope (double) - the slope parameter value (defaults to 1)
// amplify (double) - the amplify parameter value (defaults to 1)

NNetUnit::NNetUnit(ActiveT activationMode, double slope, double amplify)
{
	mInput = -1;
	mActivationType = activationMode;

	// ignore invalid values
	if(slope > 0)
	{
		mSlope = slope;
	}

	// ignore invalid values
	if(amplify > 0)
	{
		mAmplify = amplify;
	}
}

/////////////////////////////////////////////////////////////////////
// destructor

NNetUnit::~NNetUnit()
{
}

/////////////////////////////////////////////////////////////////////
// Public Methods
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// setSlope - sets the slope parameter of the activation function
//
// This parameter can be used to adjust the sensitivity of the 
// activation function (see above).
// 
// slope (double) - the slope parameter value

void NNetUnit::setSlope(double slope)
{
	// ignore invalid values
	if(slope > 0)
	{
		mSlope = slope;
	}
}

/////////////////////////////////////////////////////////////////////
// setAmplify - sets the amplify parameter of the activation function
//
// This parameter can be used to increase (amplify > 1) or decrease 
// (amplify < 1) the activation value which alters the range of the 
// function.
//
// amplify (double) - the amplify parameter value

void NNetUnit::setAmplify(double amplify)
{
	// ignore invalid values
	if(amplify > 0)
	{
		mAmplify = amplify;
	}
}

/////////////////////////////////////////////////////////////////////
// getActivation - gets the activation value of the neuron
//
// returns the activation value

double NNetUnit::getActivation()
{
	double activation = 0;

	switch(mActivationType)
	{
	case kThreshold:    // default range: 0 OR mSlope
                        // amplified range: 0 OR (mSlope * mAmplify)
		if(mInput >= 0)
		{
			activation = 1 * mSlope;
		}
		break;

	case kUnipolar:     // default range: 0 to 1
                        // amplified range: 0 to mAmplify

		activation = 1.0 / (1.0 + exp(-mSlope * mInput));
		break;

	case kBipolar:      // default range: -1 to 1
                        // amplified range: -mAmplify to mAmplify

		activation = (2.0 / (1.0 + exp(-mSlope * mInput))) - 1;
		break;

	case kTanh:         // default range: -1 to 1
                        // amplified range: -mAmplify to mAmplify

		activation = tanh(mSlope * mInput);
		break;

	case kGauss:        // default range: 0 to 1
                        // amplified range: 0 to mAmplify

		activation = exp(-mSlope * mInput * mInput);
		break;

	case kArctan:       // default range: -pi/2 to +pi/2
                        // amplified range: -(pi/2) * mAmplify to +(pi/2) * mAmplify

		activation = atan(mSlope * mInput);
		break;

	case kSin:          // default range: -1 to 1
                        // amplified range: -mAmplify to mAmplify

		activation = sin(mSlope * mInput);
		break;

	case kCos:          // default range: -1 to 1
                        // amplified range: -mAmplify to +mAmplify

		activation = cos(mSlope * mInput);
		break;

	case kSinC:         // default range: ~ -0.217234 to 1
                        // amplified range: ~ -(mAmplify * 0.217234) to mAmplify

		if(fabs(mInput) < 0.00001)
		{
			activation = 1.0;
		}
		else
		{
			activation = sin(mSlope * mInput) / (mSlope * mInput);
		}

		break;

	case kElliot:       // default range: 0 to 1
                        // amplified range: 0 to mAmplify

		activation = ((mSlope * mInput) / 2) / (1 + fabs(mSlope * mInput)) + 0.5;
		break;

	case kLinear:       // range: -infinity to +infinity
						
		activation = mSlope * mInput;
		break;

	case kISRU:         // default range: -1 / sqrt(mSlope) to 1 / sqrt(mSlope)
                        // amplified range: -(mAmplify / sqrt(mSlope)) to +(mAmplify / sqrt(mSlope))

		activation = mInput / sqrt(1 + mSlope * mInput * mInput);
		break;

	case kSoftSign:		// default range: -1 to 1
						// amplified range: -mAmplify to mAmplify

		activation = (mSlope * mInput) / (1 + fabs(mSlope * mInput));
		break;

	case kSoftPlus:     // range: 0 to +infinity

		activation = log(1 + exp(mSlope * mInput));
		break;
	}

    // the activation value is increased if amplify > 1 or reduced if amplify < 1
	return mAmplify * activation;
}

/////////////////////////////////////////////////////////////////////
