// Standard Libraries
#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>

// Parameters
#include "Parameters.h"

// Utilities
#include "utilities/include/PrintVector.h"
#include "utilities/include/Validate.h"
#include "utilities/include/WarmupGPU.h"

// Solver factory
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
	cout << "Module 5 - HW 1" << endl;
	cout << "Vector Size: " << N << endl;
	
	// Print vector object
//	PrintVector<Real> printVector;

	// Validate object
	Validate<Real> validate;

	// Vector A - filled with 1s
	vector<Real> A(N, 1.0);

	// Resulting reference & test vectors
	vector<Real> refSum(1, 0.0);
	vector<Real> testSum(1, 0.0);

	WarmupGPU warmupGPU;
	warmupGPU.setup(refGPU, testGPU);

	cout << "RefGPU = " << refGPU << endl;
	cout << "TestGPU = " << testGPU << endl;

	// Reference Solver
	cout << "\nSolver: " << refSolverName << endl;
	SolverFactory<Real> refSolverFactory(A, refSum);
	std::shared_ptr<ISolver<Real>> refSolver = refSolverFactory.getSolver(refSolverName);
	if (refGPU)
	{	
		warmupGPU.warmup();	
		cout << "Warmup for reference solver: " << refSolverName << endl;
	}
	auto tInit = omp_get_wtime();
	refSolver->solver();
	auto tFin = omp_get_wtime();
	auto runtimeRef = (tFin - tInit) * 1000.0; // in ms
	if (validate.validate(refSum))
	{
		cout << "Ref sums correct!" << endl;
	}
	else
	{
		cout << "Ref sums NOT correct!" << endl;
	}
	
	// Test Solver
	cout << "\nSolver: " << testSolverName << endl;
	SolverFactory<Real> testSolverFactory(A, testSum);
	std::shared_ptr<ISolver<Real>> testSolver = testSolverFactory.getSolver(testSolverName);
	if ((!refGPU) && testGPU)
	{
		warmupGPU.warmup();
		cout << "Warmup for test solver: " << testSolverName << endl;
	}
	tInit = omp_get_wtime();
	testSolver->solver();
	tFin = omp_get_wtime();
	auto runtimeTest = (tFin - tInit) * 1000.0; // in ms
	if (validate.validate(testSum))
	{
		cout << "Test sums correct!" << endl;
	}
	else
	{
		cout << "Test sums NOT correct!" << endl;
	}

	// Testing purposes
	/*
	printVector.printVector(A);
	printVector.printVector(refSum);
	printVector.printVector(testSum);
	*/

	cout << "\nRuntimes: " << endl;
	cout << std::setprecision(2) << std::fixed;
	cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
	cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
	cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;

}