/////////////////////////////////////////////////////////////////////
//
// Model Fit Test Application
//
// Author: Jason Jenkins
//
// This application provides three examples of using a neural network
// to model the potential relationship between a single response Y
// and a single predictor variable X. The three examples are:
//
// 1) The relationship between 'horsepower' (X) and 'mpg' (Y)
//    in the 'Auto' database.
//
// 2) The relationship between 'Balance' (X) and 'Rating' (Y)
//    in the 'Credit' database.
//
// 3) The relationship between 'age' (X) and 'wage' (Y)
//    in the 'Wage' database.
//
// The three data sets are provided in .CSV form and this application
// assumes that these files are placed in the same working directory
// as the executable file built from this source. All output files
// are also written to this same directory. In all cases the output
// consists of the trained neural network in serialised form written
// to a file named TrainedNetModelFit#.dat, the trained model output
// data X and Y values written to a file named TrainModelOutput#.csv
// and the model validation output X and Y values written to a file
// named ValidModelOutput#.csv (where # = 1, 2 or 3 depending on the 
// selected example).  The generated .CSV files can then be loaded 
// into a spreadsheet, such as Excel, and plotted on the same chart
// as the relevant X- and Y-values from the database file to observe
// how the model compares to the data.
//
// These three data sets and various model fits are discussed in
// detail in the book "An Introduction to Statistical Learning with
// Applications in R" by Gareth James, Daniela Witten, Trevor Hastie
// and Robert Tibshirani published by Springer.
//
/////////////////////////////////////////////////////////////////////

#include "NeuralNet.h"
#include "NNetTrainer.h"
#include "DataTable.h"

/////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>

/////////////////////////////////////////////////////////////////////
// The maximum floating point value

const double FPMAX = numeric_limits<double>::max();

/////////////////////////////////////////////////////////////////////
// Test Procedures
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/// <summary>
/// creates the neural network training set - 
/// 
/// The input and target vectors are extracted from the database
/// file and stored within vectors so that they can be used by the 
/// neural network training routine.  In these examples both the
/// input and target vectors contain only single values. The values
/// are also scaled so that they are less than 1 in magnitude - this
/// helps the network training process.
/// </summary>
/// <param name="fName">the name of the file containing the training data</param>
/// <param name="xCol">the name of the column representing the x-values</param>
/// <param name="yCol">the name of the column representing the y-values</param>
/// <param name="scale">used to scale the data to make all values less than 1</param>
/// <param name="inVecs">the training set input vectors</param>
/// <param name="targetVecs">the training set target vectors</param>
/// <returns>-1 if there is a problem loading the database file otherwise 0</returns>
/// 
int createTrainingSet(const string& fName, const string& xCol, const string& yCol, double scale,
					  vector<vector<double> >& inVecs, vector<vector<double> >& targetVecs)
{
	int retVal = -1;

	// we are only using single input and target values in these examples but 
	// the neural net allows for multiple input and target values using vectors
	// so the single x- and y-values are stored as vectors with only one element
	vector<double> iVec, tVec, dX, dY;

	// create a data table from the database file containing the training data
	DataTable dataTab(fName);

	// check that the data has been read from the file without any errors
	if(dataTab.getNumRows() > 0 && dataTab.getNumCols() > 0)
	{
		// read the x-input and y-target values from the data table
		dataTab.getNumericCol(xCol, dX);
		dataTab.getNumericCol(yCol, dY);

		int size = (int)dX.size();

		// populate the training set input and target vectors
		for(int i = 0; i < size; i++)
		{
			// scale the input and output values so their magnitudes are less than 1
			iVec.push_back(dX[i] * scale);	// training set input vector
			tVec.push_back(dY[i] * scale);	// training set target vector

			// the input and target vectors are also stored within a vector
			inVecs.push_back(iVec);
			targetVecs.push_back(tVec);

			iVec.clear();
			tVec.clear();
		}

		retVal = 0;
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// trains the neural network - 
/// 
/// Trains the supplied neural network using the given learning
/// constant, momentum term (this can be zero), input and target
/// vectors.  Training continues until the maximum number of
/// iterations has been exceeded or the total network error is
/// less than the supplied minimum network error value.  In the
/// former case the trained neural network is the network that
/// achieved the minimum network error.
/// </summary>
/// <param name="net">the neural network to be trained</param>
/// <param name="learnConst">the learning constant</param>
/// <param name="momentum">the momentum term</param>
/// <param name="minNetError">the minimum network error</param>
/// <param name="nIterations">the maximum number of iterations</param>
/// <param name="inVecs">the training set input vectors</param>
/// <param name="targetVecs">the training set target vectors</param>
/// 
void trainNNet(NeuralNet& net, double learnConst, double momentum, double minNetError, int nIterations,
			   const vector<vector<double> >& inVecs, const vector<vector<double> >& targetVecs)
{
	bool converged = false;
	double netError, minErr = FPMAX;
	NeuralNet minNet;       // keeps track of the network with the minimum network error
	NNetTrainer trainer;    // this object trains the neural net

	// initialize the trainer
	trainer.addNewTrainingSet(inVecs, targetVecs);
	trainer.setLearningConstant(learnConst);
	trainer.setMomentum(momentum);	

	// carry out the training
	cout << "Training Started - Please wait..." << endl;

	for(int i = 1; i <= nIterations; i++)
	{
		trainer.trainNeuralNet(net);
		netError = trainer.getNetError();

		if(netError < minNetError)
		{
			cout << "The solution has converged after " << i << " iterations!" << endl;
			cout << "Target Minimum Network Error: " << minNetError << endl;
			cout << "Current Network Error: " << netError << endl;
			cout << "This neural network will be used to fit the model." << endl;
			converged = true;
			break;
		}

		// keep track of the minimum error value
		if(netError < minErr)
		{
			// copy the state of the neural net at the minimum error value
			minNet = net;
			minErr = netError;
		}

		// show current progress
		cout << "Iterations: " << i << " - Net Error: " << netError << " - Min Error: " << minErr << endl;

		trainer.resetNetError();
	}

	if(!converged)
	{
		cout << "The solution has not converged - The Network Error is still above the target: " << minNetError << endl;
		cout << "Current Network Error: " << netError << endl;
		cout << "The minimum error value was: " << minErr << endl;
		cout << "The neural network that achieved this minimum will be used to fit the model." << endl;

		// return the net with settings at the minimum error value
		net = minNet;
	}

	cout << "Training Complete!" << endl;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// outputs the input x-values and the trained y-values to a file - 
/// 
/// The supplied neural network generates response values corresponding
/// to the given set of input vectors. In these examples both the input
/// and output vectors contain only single values and this procedure
/// makes the assumption that the supplied network and input vectors
/// correspond to this format.  The same scale factor used to create
/// the training set should also be supplied so that the output data
/// is the same scale as the original data file.  The re-scaled input
/// x-values and response y-values are written to a file with the given
/// name.
/// </summary>
/// <param name="net">the trained neural network</param>
/// <param name="inVecs">the training set input vectors</param>
/// <param name="scale">used to scale the data back to its original size</param>
/// <param name="fname">the name of the file the data will be written to</param>
/// 
void generateTrainedOutput(NeuralNet& net, const vector<vector<double> >& inVecs, double scale, const string& fname)
{
	ofstream outFile(fname);
	vector<double> dX, dY;
	int size = (int)inVecs.size();

	if(outFile.good())
	{
		// set the output precision for double values
		outFile << std::setprecision(16);

		// output the column titles
		outFile << "X, Y" << endl;

		for(int i = 0; i < size; i++)
		{
			dX = inVecs[i];

			// calculate the response y-values given the input x-values from the training set
			net.getResponse(dX, dY);

			// the required values are stored in vectors and need re-scaling
			outFile << dX[0] / scale << "," << dY[0] / scale << endl;
		}

		outFile.close();

		cout << "The trained output has been written to the file: " << fname << endl;
	}
	else
	{
		cout << "ERROR: Writing to file - unable to open or create the file: " << fname << endl;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// helper procedure to train the neural network - 
/// 
/// This procedure calls the procedures to create the training set,
/// train the neural network and write the trained responses to a file.
/// </summary>
/// <param name="net">the neural network to be trained</param>
/// <param name="learnConst">the learning constant</param>
/// <param name="momentum">the momentum term</param>
/// <param name="minNetError">the minimum network error</param>
/// <param name="nIterations">the maximum number of iterations</param>
/// <param name="scaleFac">used to scale the data to make all values less than 1</param>
/// <param name="dataFile">the name of the file containing the training data</param>
/// <param name="xCol">the name of the column representing the x-values</param>
/// <param name="yCol">the name of the column representing the y-values</param>
/// <param name="outputFile">the name of the file the trained data will be written to</param>
///
/// <returns>-1 if there is a problem loading the training set otherwise 0</returns>
/// 
int carryOutTraining(NeuralNet& net, double learnConst, double momentum, 
					  double minNetError, int nIterations, double scaleFac,
					  const string& dataFile, const string& xCol, const string& yCol,
					  const string& outputFile)
{
	int retVal = -1;

	vector<vector<double> > inVecs;         // a vector of input vectors
	vector<vector<double> > targetVecs;     // a vector of target vectors

	// get and create the training set data
	if(createTrainingSet(dataFile, xCol, yCol, scaleFac, inVecs, targetVecs) == 0)
	{
		// train the neural network
		trainNNet(net, learnConst, momentum, minNetError, nIterations, inVecs, targetVecs);

		// write to file the output generated by the training set input values and the trained net
		generateTrainedOutput(net, inVecs, scaleFac, outputFile);

		retVal = 0;
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// generates responses to input values not part of the training set 
/// and writes this to a file - 
/// 
/// The trained neural network is built from its file representation
/// and responses are generated from input values that are largely
/// (though not exclusively) absent from the training set. In these
/// examples both the input and output vectors contain only single
/// values and this procedure makes the assumption that the supplied
/// network and input vectors correspond to this format.  The same 
/// scale factor used to create the training set should also be 
/// supplied so that the output data is the same scale as the
/// original data file.  The validation set x-values and response
/// y-values are written to a file with the given name.
/// </summary>
/// <param name="xStart">the starting x-value of the validation set</param>
/// <param name="xEnd">the ending x-value of the validation set</param>
/// <param name="xStep">the step-size of the validation set</param>
/// <param name="scale">used to scale the data back to its original size</param>
/// <param name="netFname">the name of the file containing the trained net</param>
/// <param name="outFname">the name of the file the data will be written to</param>
/// 
void validateModelFit(double xStart, double xEnd, double xStep, double scale, const string& netFname, const string& outFname)
{
	NeuralNet net(netFname);		// construct the neural network from a data file
	ofstream outFile(outFname);
	vector<double> dX, dY;

	if(outFile.good())
	{
		// set the output precision for double values
		outFile << std::setprecision(16);

		// output the column titles
		outFile << "X, Y" << endl;

		for(double xVal = xStart; xVal <= xEnd; xVal += xStep)
		{
			dX.push_back(xVal * scale);

			// calculate the response y-values given the input x-values from the validation set
			net.getResponse(dX, dY);

			// the required values are stored in vectors and need re-scaling
			outFile << dX[0] / scale << "," << dY[0] / scale << endl;

			dX.clear();
		}

		outFile.close();

		cout << "The validation set data has been written to the file: " << outFname << endl;
	}
	else
	{
		cout << "ERROR: Writing to file - unable to open or create the file: " << outFname << endl;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// models the relationship between 'horsepower' and 'mpg' in the 
/// 'Auto' database
///
/// With the current settings the network is trained in only 6 
/// iterations (compiled under MSVC 10). Try changing the values of
/// outSlope, hiddenSlope and hiddenAmplify to see how they effect
/// the number of iterations and compare the different solutions on
/// the same plot as the training set to see how they fit the data.
/// Try out the different activation functions as well.
///
/// N.B. leave outAmplify set to 1 so that the network output values
/// are in the correct range. The training set data is scaled so that
/// all values are less than 1 in magnitude (to help the training 
/// process). The network output values need to be in the range 0 - 1
/// so thet they can be re-scaled to match the data in the original
/// training set.
/// </summary>
/// 
void modelFitExample1()
{
	string sDbaseFile = "Auto.csv";                 // the input file containing the database
	string sXcol = "horsepower";                    // the name of the column containing the x-values
	string sYcol = "mpg";                           // the name of the column containing the y-values
	string sTrainOutF = "TrainModelOutput1.csv";    // the output file for the input x-values and trained y-values
	string sTrainNetF = "TrainedNetModelFit1.dat";  // the trained net is written to this file
	string sValidOutF = "ValidModelOutput1.csv";    // the output file for the validation set x- and y-values

	// This value is used to set the range of the initial random values
	// for the weighted connections linking the layers of the neural net.
	// 1 represents the range -0.5 to +0.5; 2 represents -1.0 to +1.0; 
	// 3 represents -1.5 to +1.5; 4 represents -2.0 to +2.0 etc.
	// N.B. randomising within the range -2 to +2 is usually sufficient.
	int initRange = 2;		// -1.0 to +1.0	

	// The learning constant governs the 'size' of the steps taken down
	// the error surface. Larger values decrease training time but can
	// lead to the system overshooting the minimum value.
	// N.B. values between 0.001 and 10 have been reported as working successfully.
	double learnConst = 0.01;
	
	// The momentum term can be used to weight the search of the error surface
	// to continue along the same 'direction' as the previous step.
	// A value of 1 will add 100% of the previous weighted connection
	// value to the next weighted connection adjustment. If set to zero the
	// next step of the search will always proceed down the steepest path
	// of the error surface.
	double momentum = 0;
		
	// The slope can be used to adjust the sensitivity of the neurons
	// activation functions - see NNetUnit.cpp for details
	double outSlope = 35;
	double hiddenSlope = 5;

	// The amplify factor can be used to boost or reduce the neurons signal
	double outAmplify = 1;      // we do not want to amplify the output signal
	double hiddenAmplify = 45;

	// The convergence criteria - training will stop when the total
	// network error is less than this value.
	// N.B. too low a value can lead to overtraining which can lock
	// the network in a final state that is difficult to generalise from.
	double minNetError = 0.005;

	// The maximum number of iterations - if a solution has not been
	// found the search will terminate once this value has been exceeded.
	int nIterations = 5000;

	// This is used to scale the training data so that it falls within the range: 0 <= x <= 1
	double scaleFac = 0.001;

	// use a fixed seed so the results can be repeated
	srand(1);

	// create the neural network
	NeuralNet net;

	// set up the input and output layers
	net.setNumInputs(1);                    // a single input value in this example (the 'x-value')
	net.setNumOutputs(1);                   // a single output value in this example (the 'y-value')
	net.setOutputUnitType(kElliot);         // the output layer activation function - see NNetUnit.cpp for details of the available types
	net.setOutputUnitSlope(outSlope);       // the slope value to be used by the output layer activation function
	net.setOutputUnitAmplify(outAmplify);   // the amplify value to be used by the output layer activation function

	// add a hidden layer of 4 neurons with the given activation function and settings
	net.addLayer(4, kISRU, initRange, hiddenSlope, hiddenAmplify);		// this will be the first layer in a multi-layer network
	
	// train the net to model the relationship between 'horsepower' and 'mpg' in the 'Auto' database
	if(carryOutTraining(net, learnConst, momentum, minNetError, nIterations, scaleFac, sDbaseFile, sXcol, sYcol, sTrainOutF) == 0)
	{
		// write the trained neural net to file - a new net can be constructed from a file for analyzing new input data
		net.writeToFile(sTrainNetF);

		// validate the model using input x-values (40 to 250 in steps of 0.25) most of which are not in the training set
		validateModelFit(40, 250, 0.25, scaleFac, sTrainNetF, sValidOutF);
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// models the relationship between 'Balance' and 'Rating' in the 
/// 'Credit' database
///
/// The Gaussian and SoftPlus activation functions both provide 
/// differen fits to the data. However, other activation types 
/// including Tanh and Sine do not fit the data very well at all.
/// </summary>
///
void modelFitExample2()
{
	string sDbaseFile = "Credit.csv";               // the input file containing the database
	string sXcol = "Balance";                       // the name of the column containing the x-values
	string sYcol = "Rating";                        // the name of the column containing the y-values
	string sTrainOutF = "TrainModelOutput2.csv";    // the output file for the input x-values and trained y-values
	string sTrainNetF = "TrainedNetModelFit2.dat";  // the trained net is written to this file
	string sValidOutF = "ValidModelOutput2.csv";    // the output file for the validation set x- and y-values

	// set the initial values for the net - see modelFitExample1() for an explanation of these terms
	int initRange = 4;			// -2.0 to +2.0
	double learnConst = 0.01;
	double momentum = 0;
	double outSlope = 5;
	double hiddenSlope = 5;
	double outAmplify = 1;      // we do not want to amplify the output signal
	double hiddenAmplify = 5;
	
	double minNetError = 0.28;
	int nIterations = 10000;
	double scaleFac = 0.0005;
	
	// use a fixed seed so the results can be repeated
	srand(3);

	// create the neural network
	NeuralNet net;	

	// set up the input and output layers
	net.setNumInputs(1);                    // a single input value in this example (the 'x-value')
	net.setNumOutputs(1);                   // a single output value in this example (the 'y-value')
	net.setOutputUnitType(kElliot);         // the output layer activation function - see NNetUnit.cpp for details of the available types
	net.setOutputUnitSlope(outSlope);       // the slope value to be used by the output layer activation function
	net.setOutputUnitAmplify(outAmplify);   // the amplify value to be used by the output layer activation function

	// add a hidden layer of 4 neurons with the given activation function and settings
	// the Gaussian and SoftPlus activation functions seem to provide the best fit for this second example (try some of the others e.g. Tanh)
	net.addLayer(4, kSoftPlus, initRange, hiddenSlope, hiddenAmplify);

	// train the net to model the relationship between 'Balance' and 'Rating' in the 'Credit' database
	if(carryOutTraining(net, learnConst, momentum, minNetError, nIterations, scaleFac, sDbaseFile, sXcol, sYcol, sTrainOutF) == 0)
	{
		// write the trained neural net to file
		net.writeToFile(sTrainNetF);

		// validate the model using input x-values (-100 to 2100 in steps of 0.5) most of which are not in the training set
		validateModelFit(-100, 2100, 0.5, scaleFac, sTrainNetF, sValidOutF);
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// models the relationship between 'age' and 'wage' in the 'Wage' 
/// database
///
/// The ISRU (Inverse square root unit), Elliot and Soft Sign 
/// activation functions all provide fits to the data but they also
/// exhibit subtle differences.
/// </summary>
///
void modelFitExample3()
{
	string sDbaseFile = "Wage.csv";                 // the input file containing the database
	string sXcol = "age";                           // the name of the column containing the x-values
	string sYcol = "wage";                          // the name of the column containing the y-values
	string sTrainOutF = "TrainModelOutput3.csv";    // the output file for the input x-values and trained y-values
	string sTrainNetF = "TrainedNetModelFit3.dat";  // the trained net is written to this file
	string sValidOutF = "ValidModelOutput3.csv";    // the output file for the validation set x- and y-values

	// set the initial values for the net - see modelFitExample1() for an explanation of these terms
	int initRange = 10;             // -5.0 to +5.0
	double learnConst = 0.01;
	double momentum = 0;
	double outSlope = 10;
	double hiddenSlope = 10;
	double outAmplify = 1;          // we do not want to amplify the output signal
	double hiddenAmplify = 10;
	double minNetError = 2.3;
	int nIterations = 7000;
	double scaleFac = 0.001;
	
	// use a fixed seed so the results can be repeated
	srand(5);

	// create the neural network
	NeuralNet net;	

	// set up the input and output layers
	net.setNumInputs(1);                    // a single input value in this example (the 'x-value')
	net.setNumOutputs(1);                   // a single output value in this example (the 'y-value')
	net.setOutputUnitType(kElliot);         // the output layer activation function - see NNetUnit.cpp for details of the available types
	net.setOutputUnitSlope(outSlope);       // the slope value to be used by the output layer activation function
	net.setOutputUnitAmplify(outAmplify);   // the amplify value to be used by the output layer activation function

	// add a hidden layer of 4 neurons with the given activation function and settings
	// the ISRU, Elliot and Soft Sign activation functions all provide (different) fits for this third example
	net.addLayer(4, kSoftSign, initRange, hiddenSlope, hiddenAmplify);

	// train the net to model the relationship between 'age' and 'wage' in the 'Wage' database
	if(carryOutTraining(net, learnConst, momentum, minNetError, nIterations, scaleFac, sDbaseFile, sXcol, sYcol, sTrainOutF) == 0)
	{
		// write the trained neural net to file
		net.writeToFile(sTrainNetF);

		// validate the model using input x-values (15 to 85 in steps of 0.1) most of which are not in the training set
		validateModelFit(15, 85, 0.1, scaleFac, sTrainNetF, sValidOutF);
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// helper procedure to run the required example
/// </summary>
/// <param name="nExample">the example to run</param>
/// 
void runExample(int nExample)
{
	switch(nExample)
	{
	case 1:
		modelFitExample1();
		break;

	case 2:
		modelFitExample2();
		break;

	case 3:
		modelFitExample3();
		break;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// provides the user with a simple menu to pick and run each of the 
/// three example model fits
/// </summary>
/// 
int main()
{	
	// provide the user with a simple menu to pick and run the example model fits
	while(true)
	{
		int num = -1;

		cout << std::endl << "Example 1: Models the relationship between 'horsepower' and 'mpg' in the" << std::endl;
		cout <<	"           'Auto' database (converges very quickly)." << std::endl << std::endl;

		cout << std::endl << "Example 2: Models the relationship between 'Balance' and 'Rating' in the" << std::endl;
		cout <<	"           'Credit' database (runs for 10,000 iterations - fairly quickly)." << std::endl << std::endl;

		cout << std::endl << "Example 3: Models the relationship between 'age' and 'wage' in the" << std::endl;
		cout <<	"           'Wage' database (runs for 7,000 iterations - fairly slowly)." << std::endl << std::endl;

		cout << "Enter the Example you wish to run (1, 2 or 3 - any other value to quit): ";
		cin >> num;
		
		if(num > 3 || num < 1)
		{
			break;
		}
		else
		{
			runExample(num);
		}
	}
}

/////////////////////////////////////////////////////////////////////
