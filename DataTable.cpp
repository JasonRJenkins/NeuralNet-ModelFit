/////////////////////////////////////////////////////////////////////
//
// Implements the Data Table class
//
// Author: Jason Jenkins
//
// This class is a representation of a database table that can be
// easily manipulated for use in mathematical or statistical analysis.
//
// A DataTable object can be created from a file representation of
// a table in .CSV format or by adding rows individually.
//
// Non-numeric values are automatically assigned a numeric alias to 
// help facilitate mathematical analysis of the data. These values
// can be overridden if desired.
//
/////////////////////////////////////////////////////////////////////

#include "DataTable.h"

/////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>

/////////////////////////////////////////////////////////////////////
/// <summary>
/// default constructor
/// </summary>
/// 
DataTable::DataTable()
{
	m_Rows = 0;
	m_Cols = 0;
	m_Header = false;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// constructs a DataTable object from a .CSV file representation
/// </summary>
/// <param name="fName">the name of the .CSV file containing the data</param>
/// <param name="header">set it to true if the data has a header row
///                      containing column names otherwise set it to false</param>
///                      
DataTable::DataTable(const string& fName, bool header)
{
	m_Rows = 0;
	m_Cols = 0;
	m_Header = header;

	// read in the data from the file stream
	readFromStream(fName, header);

	// initialise the alias vector values to zero
	for(int i = 0; i < m_Cols; i++)
	{
		m_AliasVec.push_back(0);
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// destructor
/// </summary>
///
DataTable::~DataTable()
{
}

/////////////////////////////////////////////////////////////////////
// Public Methods
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/// <summary>
/// clears a DataTable object ready for re-use
/// </summary>
/// 
void DataTable::clearTable()
{
	m_Rows = 0;
	m_Cols = 0;
	m_Header = false;

	m_ColumnNames.clear();
	m_RawData.clear();
	m_ColIdx.clear();
	m_Aliases.clear();
	m_AliasVec.clear();
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// adds a new row to the DataTable
/// </summary>
/// <param name="row">a vector of column values in string format</param>
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::addRawRow(const vector<string>& row)
{
	int retVal = 0;

	// check for the very first row to be added
	if(m_Rows <= 0 && m_Cols <= 0)
	{
		m_Rows = 0;
		m_Cols = (int)row.size();

		// initialise the alias vector values to zero
		for(int i = 0; i < m_Cols; i++)
		{
			// this keeps track of the automatic alias values used by each column
			m_AliasVec.push_back(0);
		}
	}

	if(m_Cols == (int)row.size())
	{
		m_RawData.push_back(row);
		m_Rows++;
	}
	else
	{
		cout << "ERROR: You cannot insert a row with: " << row.size();
		cout << " columns into a table with: " << m_Cols << " columns!" << endl;
		retVal = -1;
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets a data row with the values in string format
/// </summary>
/// <param name="nRow">the numeric index of the row to be returned</param>
/// <param name="row">the row data as a vector of string values</param>
///
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::getRawRow(int nRow, vector<string>& row)
{
	int retVal = 0;

	if((nRow < 0) || ((nRow + 1) > m_Rows))
	{
		cout << "ERROR: The requested row index: " << nRow;
		cout << " is out of bounds!" << endl;
		cout << "INFORMATION: The table has: " << m_Rows;
		cout << " rows, indexed from 0 to " << (m_Rows - 1) << "." << endl;
		retVal = -1;
	}
	else
	{
		row = m_RawData[nRow];
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets a data row with the values in double format - 
/// 
/// Non-numeric data is automatically assigned a numeric alias if an
/// alias has not already been set up. The first non-numeric entry in 
/// a column is assigned the value 0 the next distinct entry is 
/// assigned the value 1 and the next 2 and so on. You can set your 
/// own alias values using the setAlias method.
/// </summary>
/// <param name="nRow">the numeric index of the row to be returned</param>
/// <param name="row">the row data as a vector of double values</param>
///
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::getNumericRow(int nRow, vector<double>& row)
{
	int retVal = 0;

	if((nRow < 0) || ((nRow + 1) > m_Rows))
	{
		cout << "ERROR: The requested row index: " << nRow;
		cout << " is out of bounds!" << endl;
		cout << "INFORMATION: The table has: " << m_Rows;
		cout << " rows, indexed from 0 to " << (m_Rows - 1) << "." << endl;
		retVal = -1;
	}
	else
	{
		// get the row data in raw (string) format
		vector<string>& rawRow = m_RawData[nRow];
		int len = (int)rawRow.size();

		// convert each row element from a string to a numeric value
		for(int i = 0; i < len; i++)
		{
			double value = 0;
			string sVal = rawRow[i];

			// check to see if we already have an alias set up
			string sCol = to_string((long double)i);
			
			// use the string column value and the column index as the alias key
			// this allows multiple aliases for the same value to be set up e.g.
			// in one column you may have values: 'red', 'blue' and 'green' aliased
			// to values 0, 1 and 2 respectively and in a second column you may have 
			// values: 'black', 'yellow' and 'red' aliased to 0, 1 and 2 respectively
			// i.e. two aliases for 'red' are set up, one for each column
			string sKey = sVal + " " + sCol;
			map<string, string>::iterator it = m_Aliases.find(sKey);

			if(it != m_Aliases.end())
			{
				// an alias has already been assigned
				sVal = it->second;
			}

			// try the string to value conversion
			try
			{
				value = stod(sVal);
			}
			catch(const std::invalid_argument& ia)
			{				
				// assign non-numeric column values integer values starting with zero
				value = m_AliasVec[i];			// this vector has initial values set to 0
				m_AliasVec[i] = value + 1;		// the next value to use

				// create and add the value-alias to the aliases map
				string strAlias = to_string((long double)value);
				m_Aliases.insert(make_pair(sKey, strAlias));
			}

			row.push_back(value);
		}
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets a data column with the values in string format
/// </summary>
/// <param name="nCol">the numeric index of the column to be returned</param>
/// <param name="col">the column data as a vector of string values</param>
///
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::getRawCol(int nCol, vector<string>& col)
{
	int retVal = 0;

	if((nCol < 0) || ((nCol + 1) > m_Cols))
	{
		cout << "ERROR: The requested column index: " << nCol;
		cout << " is out of bounds!" << endl;
		cout << "INFORMATION: The table has: " << m_Cols;
		cout << " columns, indexed from 0 to " << (m_Cols - 1) << "." << endl;
		retVal = -1;
	}
	else
	{
		// extract the column data
		for(int rows = 0; rows < m_Rows; rows++)
		{
			col.push_back(m_RawData[rows][nCol]);				
		}
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets a data column with the values in string format
/// </summary>
/// <param name="sCol">the name of the column to be returned</param>
/// <param name="col">the column data as a vector of string values</param>
///
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::getRawCol(const string& sCol, vector<string>& col)
{
	int iCol;
	int retVal = 0;

	// map the string "index" to the actual integer index
	map<string, int>::iterator it = m_ColIdx.find(sCol);

	if(it != m_ColIdx.end())
	{
		iCol = it->second;

		// use the numeric indexed version of getRawCol
		retVal = getRawCol(iCol, col);
	}
	else
	{
		cout << "ERROR: The requested column index string: " << sCol;
		cout << " does not have a corresponding numeric index assigned to it!" << endl;
		retVal = -1;
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets a data column with the values in double format - 
/// 
/// Non-numeric data is automatically assigned a numeric alias if an
/// alias has not already been set up. The first non-numeric entry in 
/// a column is assigned the value 0 the next distinct entry is 
/// assigned the value 1 and the next 2 and so on. You can set your 
/// own alias values using the setAlias method.
/// </summary>
/// <param name="nCol">the numeric index of the column to be returned</param>
/// <param name="col">the column data as a vector of double values</param>
///
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::getNumericCol(int nCol, vector<double>& col)
{
	int retVal = 0;

	if((nCol < 0) || ((nCol + 1) > m_Cols))
	{
		cout << "ERROR: The requested column index: " << nCol;
		cout << " is out of bounds!" << endl;
		cout << "INFORMATION: The table has: " << m_Cols;
		cout << " columns, indexed from 0 to " << (m_Cols - 1) << "." << endl;
		retVal = -1;
	}
	else
	{
		for(int rows = 0; rows < m_Rows; rows++)
		{
			double value = 0;
			string sVal = m_RawData[rows][nCol];

			// check to see if we have an alias set up
			string sCol = to_string((long double)nCol);

			// use the string column value and the column index as the alias key
			// this allows multiple aliases for the same value to be set up e.g.
			// in one column you may have values: 'red', 'blue' and 'green' aliased
			// to values 0, 1 and 2 respectively and in a second column you may have 
			// values: 'black', 'yellow' and 'red' aliased to 0, 1 and 2 respectively
			// i.e. two aliases for 'red' are set up, one for each column
			string sKey = sVal + " " + sCol;
			map<string, string>::iterator it = m_Aliases.find(sKey);

			if(it != m_Aliases.end())
			{
				// an alias has already been assigned
				sVal = it->second;
			}

			// try the string to value conversion
			try
			{
				value = stod(sVal);
			}
			catch(const std::invalid_argument& ia)
			{					
				// assign non-numeric column values integer values starting with zero
				value = m_AliasVec[nCol];			// this vector has initial values set to 0
				m_AliasVec[nCol] = value + 1;		// the next value to use

				// create and add the value-alias to the aliases map
				string strAlias = to_string((long double)value);
				m_Aliases.insert(make_pair(sKey, strAlias));
			}

			col.push_back(value);
		}
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets a data column with the values in double format
/// </summary>
/// <param name="sCol">the name of the column to be returned</param>
/// <param name="col">the column data as a vector of double values</param>
///
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::getNumericCol(const string& sCol, vector<double>& col)
{
	int iCol;
	int retVal = 0;

	// map the string "index" to the actual integer index
	map<string, int>::iterator it = m_ColIdx.find(sCol);

	if(it != m_ColIdx.end())
	{
		iCol = it->second;

		// use the numeric indexed version of getNumericCol
		retVal = getNumericCol(iCol, col);
	}
	else
	{
		cout << "ERROR: The requested column index string: " << sCol;
		cout << " does not have a corresponding numeric index assigned to it!" << endl;
		retVal = -1;
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets a numeric alias for a given string value and column index
/// </summary>
/// <param name="sValue">the value to be given an alias</param>
/// <param name="dAlias">the alias to be applied</param>
/// <param name="nCol">the column index of the column containing the value</param>
/// 
void DataTable::setAlias(const string& sValue, double dAlias, int nCol)
{
	string sAlias = to_string((long double)dAlias);

	// check to see if we have an alias set up
	string sCol = to_string((long double)nCol);

	// using the string value and the column index as the alias key
	// allows multiple aliases for the same value to be set up e.g.
	// in one column you may have values: 'red', 'blue' and 'green' aliased
	// to values 0, 1 and 2 respectively and in a second column you may have 
	// values: 'black', 'yellow' and 'red' aliased to 0, 1 and 2 respectively
	// i.e. two aliases for 'red' are set up, one for each column
	string sKey = sValue + " " + sCol;
	map<string, string>::iterator it = m_Aliases.find(sKey);

	if(it != m_Aliases.end())
	{
		// replace the pre-existing alias
		it->second = sAlias;
	}
	else
	{
		m_Aliases.insert(make_pair(sKey, sAlias));
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// returns the numeric alias for a given string value and column index
/// </summary>
/// <param name="sValue">the string value whose alias is required</param>
/// <param name="nCol">the column index of the column containing the value</param>
///
/// <returns>the alias value if it exists otherwise -1e10</returns>
/// 
double DataTable::getAliasValue(const string& sValue, int nCol)
{
	double value = -1e10;

	string sCol = to_string((long double)nCol);

	// using the string value and the column index as the alias key
	// allows multiple aliases for the same value to be set up e.g.
	// in one column you may have values: 'red', 'blue' and 'green' aliased
	// to values 0, 1 and 2 respectively and in a second column you may have 
	// values: 'black', 'yellow' and 'red' aliased to 0, 1 and 2 respectively
	// i.e. two aliases for 'red' are set up, one for each column
	string sKey = sValue + " " + sCol;
	map<string, string>::iterator it = m_Aliases.find(sKey);

	if(it != m_Aliases.end())
	{
		string sVal = it->second;

		// try the string to value conversion
		try
		{
			value = stod(sVal);
		}
		catch(const std::invalid_argument& ia)
		{
			// the alias has been set up incorrectly
			cout << "ERROR: The alias value: " << sVal << " for the string: " << sValue;
			cout << " cannot be converted to a numeric value!" << endl;
		}
	}
	else
	{
			// an alias has not been set up
			cout << "ERROR: The string: " << sValue << " does not have an alias!" << endl;
	}

	return value;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// returns the column index for a given named column
/// </summary>
/// <param name="sColName">the column name</param>
///
/// <returns>the integer index of the named column or -1 if it doesn't exist</returns>
/// 
int DataTable::getColIndex(const string& sColName)
{
	int iCol = -1;

	// map the string "index" to the actual integer index
	map<string, int>::iterator it = m_ColIdx.find(sColName);

	if(it != m_ColIdx.end())
	{
		iCol = it->second;
	}
	
	return iCol;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// gets the column names vector
/// </summary>
/// <param name="colNames">the vector containing the column names</param>
/// 
void DataTable::getColumnNames(vector<string>& colNames)
{
	if(m_Header == true)
	{
		colNames = m_ColumnNames;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// sets the column names vector
/// </summary>
/// <param name="colNames">the vector containing the column names</param>
/// 
void DataTable::setColumnNames(const vector<string>& colNames)
{
	int size = (int)colNames.size();

	if(m_Cols <= 0)
	{
		// we have an empty table
		m_ColumnNames = colNames;
		m_Cols = size;
		m_Header = true;
	}
	else if(m_Cols == size)
	{
		// a straight replacement for the column names
		m_ColumnNames = colNames;
		m_Header = true;
	}
	else
	{
		cout << "ERROR: The number of supplied column names: " << size;
		cout << " is not compatible with the number of columns in the table: " << m_Cols << "!" << endl;
	}
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// write this DataTable object to a .CSV file
/// </summary>
/// <param name="fName">the name of the file to write the data to</param>
///
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::writeToFile(const string& fName)
{
	ofstream outFile(fName);

	if(outFile.good())
	{
		writeToStream(outFile);
	}
	else
	{
		cout << "ERROR: Writing to file - unable to open or create the file: " << fName << endl;

		return -1;
	}

	return 0;
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// clears and re-instantiates this DataTable object from a .CSV file representation
/// </summary>
/// <param name="fName">the name of the .CSV file containing the data</param>
/// <param name="header">true if the data has a header row containing
///                      column names otherwise set to false</param>
///
/// <returns>0 if successful otherwise -1</returns>
/// 
int DataTable::readFromFile(const string& fName, bool header)
{
	int retVal = 0;

	clearTable();

	readFromStream(fName, header);

	// initialise the alias vector values to zero
	for(int i = 0; i < m_Cols; i++)
	{
		m_AliasVec.push_back(0);
	}

	// check to see if there was an error reading the file
	if(m_Rows < 0 || m_Cols < 0)
	{
		retVal = -1;
	}

	return retVal;
}

/////////////////////////////////////////////////////////////////////
// Private Methods
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/// <summary>
/// writes out the table data to a file stream buffer
/// </summary>
/// <param name="ofile">the output file stream buffer</param>
/// 
void DataTable::writeToStream(ofstream& ofile)
{
	// output the header
	if(m_Header == true)
	{
		int size = (int)m_ColumnNames.size();

		for(int i = 0; i < (size - 1); i++)
		{
			// the data is comma delimited
			ofile << m_ColumnNames[i] << ", ";
		}

		ofile << m_ColumnNames[size - 1] << endl;
	}

	// output the data
	for(int row = 0; row < m_Rows; row++)
	{
		for(int col = 0; col < m_Cols; col++)
		{
			ofile << m_RawData[row][col];

			if(col < (m_Cols - 1))
			{
				// the data is comma delimited
				ofile << ", ";
			}
		}

		ofile << endl;
	}
  	
	ofile.close();
}

/////////////////////////////////////////////////////////////////////
/// <summary>
/// reads in the table data from a file stream buffer
/// </summary>
/// <param name="fName">the name of the file to read the data from</param>
/// <param name="header">set to true if the input file has a header row 
///                      containing column names otherwise set to false</param>
///                      
void DataTable::readFromStream(const string& fName, bool header)
{	
	string line;
	ifstream myfile(fName);
	const char* t = " \t\n\r\f\v";	// white space characters

	if(myfile.is_open())
	{
		// read in the header data if it is available
		if(header == true)
		{
			if(getline(myfile, line))
			{
				bool nextLine = false;
				int idx = 0;
				int sLen = (int)line.length();
				
				if(sLen > 0)
				{		
					string sValue, sChar;

					while(!nextLine)
					{
						bool nextValue = false;

						while(!nextValue)
						{
							sChar = line[idx];

							// the data is comma delimited
							if(sChar == ",")
							{
								nextValue = true;
							}
							else if(sChar != "\"")
							{
								// quotes can be used to delimit strings - so ignore them
								sValue += sChar;
							}

							idx++;

							if(idx >= sLen)
							{
								nextLine = true;
								nextValue = true;
							}
						}

						// trim leading and trailing whitespace
						if(!sValue.empty())
						{
							sValue.erase(0, sValue.find_first_not_of(t));		// left trim
							sValue.erase(sValue.find_last_not_of(t) + 1);		// right trim
						}

						m_ColumnNames.push_back(sValue);
						m_ColIdx.insert(make_pair(sValue, m_Cols));
						m_Cols++;
						
						sValue.clear();
					}
				}
			}
		}

		// read in the data
		while(getline(myfile, line))
		{			
			vector<string> row;

			bool nextLine = false;
			int idx = 0;
			int sLen = (int)line.length();
				
			if(sLen > 0)
			{		
				bool discardLine = false;
				string sValue, sChar;

				while(!nextLine)
				{
					bool nextValue = false;

					while(!nextValue)
					{
						sChar = line[idx];

						// the data is comma delimited
						if(sChar == ",")
						{
							nextValue = true;
						}
						else if(sChar != "\"")
						{
							// quotes can be used to delimit strings - so ignore them
							sValue += sChar;
						}

						idx++;

						// check for end of line
						if(idx >= sLen)
						{
							nextLine = true;
							nextValue = true;
						}
					}

					// trim leading and trailing whitespace
					if(!sValue.empty())
					{
						sValue.erase(0, sValue.find_first_not_of(t));		// left trim
						sValue.erase(sValue.find_last_not_of(t) + 1);		// right trim
					}

					// reject data rows with missing data - identified by "?"
					if(sValue != "?")
					{
						row.push_back(sValue);
						sValue.clear();
					}
					else
					{
						discardLine = true;
						nextLine = true;
					}
				}
			
				if(!discardLine)
				{
					m_RawData.push_back(row);
					m_Rows++;
				}

				// if a header is not supplied set the number of columns
				if(m_Cols == 0 && header == false && !discardLine)
				{
					m_Cols = (int)row.size();
				}

				// check that the row sizes are consistent
				if(m_Cols != (int)row.size() && !discardLine)
				{
					cout << "ERROR: Reading from file - the data in the file: " << fName;
					cout << " does not maintain a consistent number of columns!" << endl;

					// set the number of rows and columns to invalid values to signify an error
					m_Rows = -1;
					m_Cols = -1;

					return;
				}
			}
		}

		myfile.close();
	}
	else
	{
		cout << "ERROR: Reading from file - unable to open the file: " << fName << endl;

		// set the number of rows and columns to invalid values to signify an error
		m_Rows = -1;
		m_Cols = -1;
	}
}

/////////////////////////////////////////////////////////////////////
