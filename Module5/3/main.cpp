#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>

#include "Parameters.h"
#include "utilities/include/PrintVector.h"
#include "utilities/include/Validate.h"
#include "solvers/factory/SolverFactory.h"
#include "solvers/interface/ISolver.h"

using std::cout;
using std::endl;
using std::vector;
using std::left;
using std::setprecision;
using std::setw;
using std::fixed;

int main()
{
	cout << "Module 4 - HW" << endl;
	cout << "Matrix Size: " << DSIZE << " x " << DSIZE << endl;
	
	// Print vector object
	PrintVector<Real> printVector;

	// Validate object
	Validate<Real> validate;

	// Matrix A - filled with 1s
	vector<Real> A(DSIZE * DSIZE, 1.0);

	// Resulting reference & test vectors
	vector<Real> refRowSums(DSIZE, 0.0);
	vector<Real> testRowSums(DSIZE, 0.0);
	vector<Real> refColSums(DSIZE, 0.0);
	vector<Real> testColSums(DSIZE, 0.0);

	// Reference Solver
	cout << "\nSolver: " << refSolverName << endl;
	SolverFactory<Real> refSolverFactory(A, refRowSums, refColSums);
	std::shared_ptr<ISolver<Real>> refSolver = refSolverFactory.getSolver(refSolverName);
	auto tInit = omp_get_wtime();
	refSolver->solver();
	auto tFin = omp_get_wtime();
	auto runtimeRef = (tFin - tInit) * 1000.0; // in ms
	if (validate.validate(refRowSums))
	{
		cout << "Row sums correct!" << endl;
	}
	else
	{
		cout << "Row sums NOT correct!" << endl;
	}
	if (validate.validate(refColSums))
	{
		cout << "Col sums correct!" << endl;
	}
	else
	{
		cout << "Col sums NOT correct!" << endl;
	}
	
	// Test Solver
	cout << "\nSolver: " << testSolverName << endl;
	SolverFactory<Real> testSolverFactory(A, testRowSums, testColSums);
	std::shared_ptr<ISolver<Real>> testSolver = testSolverFactory.getSolver(testSolverName);
	tInit = omp_get_wtime();
	testSolver->solver();
	tFin = omp_get_wtime();
	auto runtimeTest = (tFin - tInit) * 1000.0; // in ms
	if (validate.validate(testRowSums))
	{
		cout << "Row sums correct!" << endl;
	}
	else
	{
		cout << "Row sums NOT correct!" << endl;
	}
	if (validate.validate(testColSums))
	{
		cout << "Col sums correct!" << endl;
	}
	else
	{
		cout << "Col sums NOT correct!" << endl;
	}

	// Testing purposes
	
	printVector.printVector(A);
	printVector.printVector(refRowSums);
	printVector.printVector(refColSums);
	
	printVector.printVector(testRowSums);
	printVector.printVector(testColSums);
	

	cout << "\nRuntimes: " << endl;
	cout << std::setprecision(2) << std::fixed;
	cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
	cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
	cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;

}