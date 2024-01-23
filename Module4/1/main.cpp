#include <iostream>
#include <vector>
#include <omp.h>

#include "Parameters.h"
#include "utilities/include/PrintVector.h"
#include "utilities/include/Validate.h"
#include "solvers/factory/SolverFactory.h"
#include "solvers/interface/ISolver.h"

using std::cout;
using std::endl;
using std::vector;

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
	refSolver->rowSums();
	if (validate.validate(refRowSums))
	{
		cout << "Row sums correct!" << endl;
	}
	else
	{
		cout << "Row sums NOT correct!" << endl;
	}	
	auto tFin = omp_get_wtime();
	auto runtimeRefRows = (tFin - tInit) * 1000.0; // in ms
	refSolver->colSums();
	if (validate.validate(refColSums))
	{
		cout << "Col sums correct!" << endl;
	}
	else
	{
		cout << "Col sums NOT correct!" << endl;
	}
	auto t0 = omp_get_wtime();
	auto runtimeRefCols = (tFin - t0) * 1000.0; // in ms

	// Test Solver
	cout << "\nSolver: " << testSolverName << endl;
	SolverFactory<Real> testSolverFactory(A, testRowSums, testColSums);
	std::shared_ptr<ISolver<Real>> testSolver = testSolverFactory.getSolver(testSolverName);
	tInit = omp_get_wtime();
	testSolver->rowSums();
	if (validate.validate(testRowSums))
	{
		cout << "Row sums correct!" << endl;
	}
	else
	{
		cout << "Row sums NOT correct!" << endl;
	}
	tFin = omp_get_wtime();
	auto runtimeTestRows = (tFin - tInit) * 1000.0; // in ms
	testSolver->colSums();
	if (validate.validate(testColSums))
	{
		cout << "Col sums correct!" << endl;
	}
	else
	{
		cout << "Col sums NOT correct!" << endl;
	}
	t0 = omp_get_wtime();
	auto runtimeTestCols = (tFin - t0) * 1000.0; // in ms

	// Testing purposes
	/*
	printVector.printVector(A);
	printVector.printVector(refRowSums);
	printVector.printVector(refColSums);
	*/
	printVector.printVector(testRowSums);
	printVector.printVector(testColSums);


}