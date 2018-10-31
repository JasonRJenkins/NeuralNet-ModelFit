/////////////////////////////////////////////////////////////////////
//
// Defines the DataTable class
//
// Author: Jason Jenkins
//
// This class is a representation of a database table that can be
// easily manipulated for use in mathematical or statistical analysis.
//
/////////////////////////////////////////////////////////////////////

#pragma once

/////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <map>

/////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////

class DataTable
{
public:
	DataTable();
	virtual ~DataTable();

	// constructs a DataTable object from a .CSV file representation
	DataTable(const string& fName, bool header = true);
	
	// returns the number of rows in the table
	int getNumRows() const { return m_Rows; }

	// returns the number of columns in the table
	int getNumCols() const { return m_Cols; }

	// clears a DataTable object ready for re-use
	void clearTable();

	// adds a new row to the DataTable - the row is a vector of (string) column values
	int addRawRow(const vector<string>& row);
	
	// gets a data row with the values in string format
	int getRawRow(int nRow, vector<string>& row);

	// gets a data row with the values in double format
	int getNumericRow(int nRow, vector<double>& row);

	// gets a data column with the values in string format
	int getRawCol(int nCol, vector<string>& col);			// indexed by position
	int getRawCol(const string& sCol, vector<string>& col);	// indexed by name

	// gets a data column with the values in double format
	int getNumericCol(int nCol, vector<double>& col);			// indexed by position
	int getNumericCol(const string& sCol, vector<double>& col);	// indexed by name

	// sets a numeric alias for a given string value and column index
	void setAlias(const string& sValue, double dAlias, int nCol);

	// returns the numeric alias for a given string value and column index
	double getAliasValue(const string& sValue, int nCol);

	// returns the column index for a given column name
	int getColIndex(const string& sColName);

	// gets the column names vector
	void getColumnNames(vector<string>& colNames);

	// sets the column names vector
	void setColumnNames(const vector<string>& colNames);

	// write this DataTable object to a .CSV file
	int writeToFile(const string& fName);

	// clears and re-instantiates this DataTable object from a .CSV file
	int readFromFile(const string& fName, bool header = true);

private:
	// writes out the table data to a file stream buffer
	void writeToStream(ofstream& ofile);

	// reads in the table data from a file stream buffer
	void readFromStream(const string& fName, bool header = true);

private:	
	int m_Rows;		// the number of rows in the table
	int m_Cols;		// the number of columns in the table
	bool m_Header;	// true if the table has a header row containing column names

	vector<string> m_ColumnNames;		// the column names (if supplied)
	
	// the raw table data - each row is a vector of column values in string format
	// and the table consists of a vector of rows
	vector<vector<string> > m_RawData;

	// keeps track of the automatic alias values used by each column
	vector<double> m_AliasVec;	

	// maps a string column name to a corresponding numeric vector index
	map<string, int> m_ColIdx;

	// maps a string name to a numeric value (in string format)
	map<string, string> m_Aliases;
};

/////////////////////////////////////////////////////////////////////
