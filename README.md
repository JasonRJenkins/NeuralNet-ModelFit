# NeuralNet-ModelFit
Implements a single hidden layer feedforward neural network and uses it to fit a model to a number of datasets.

This project contains the implementation of a single hidden layer feedforward neural network.  The implementation is demonstrated by using a neural network to model the potential relationship between a single response Y and a single predictor variable X in three example datasets that can be selected by the user. The three examples are:

 1) The relationship between 'horsepower' (X) and 'mpg' (Y) in the 'Auto' database.

 2) The relationship between 'Balance' (X) and 'Rating' (Y) in the 'Credit' database.

 3) The relationship between 'age' (X) and 'wage' (Y) in the 'Wage' database.

The three data sets are provided in .CSV form and this application assumes that these files are placed in the same working directory as the executable file built from this source. All the output files are also written to this same directory. In all cases the output consists of the trained neural network in serialised form written to a file named TrainedNetModelFit#.dat, the trained model output data X and Y values written to a file named TrainModelOutput#.csv and the model validation output X and Y values written to a file named ValidModelOutput#.csv (where # = 1, 2 or 3 depending on the selected example).  The generated TrainModelOutput#.csv files can then be loaded into a spreadsheet, such as Excel, and plotted as a scatter graph to observe how the model compares to the data.

These three data sets and various model fits are discussed in detail in the book "An Introduction to Statistical Learning with Applications in R" by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani published by Springer.

To build the test application you just need to create a new empty project, which you can name ModelFit (or any other name if you like) and add all the source files to it: DataTable.cpp/.h; NeuralNet.cpp/.h; NNetTrainer.cpp/.h; NNetUnit.cpp/.h; NNetWeightedConnect.cpp/.h and TestModelFit.cpp.

The data files: Auto.csv; Credit.csv and Wage.csv should be placed in the same directory as the executable file built from this source or in the same directory as the source code if you are running the application from within Microsoft Visual Studio.

This application has been built and tested with Microsoft Visual Studio 2010 and 2017 as both a 32 and 64 bit application.  The application has also been built and tested with Cygwin GCC version 6.4.0.
