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
#include "utilities/include/RandomVectorGenerator.h"

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
	cout << "Module 5 - HW 2" << endl;
	cout << "Vector Size: " << N << endl;
	
	// Print vector object
	 PrintVector<Real> printVector;

	// Random vector generator
	RandomVectorGenerator<Real> randomVector;

	// Validate object
	Validate<Real> validate;

	// Vector A - filled with 1s
	vector<Real> A(N);
	randomVector.randomVector(A);

	// Resulting reference & test vectors
	vector<Real> refMax(1, 0.0);
	vector<Real> testMax(1, 0.0);

	WarmupGPU warmupGPU;
	warmupGPU.setup(refGPU, testGPU);

	cout << "RefGPU = " << refGPU << endl;
	cout << "TestGPU = " << testGPU << endl;

	// Reference Solver
	cout << "\nSolver: " << refSolverName << endl;
	SolverFactory<Real> refSolverFactory(A, refMax);
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

	// Test Solver
	cout << "\nSolver: " << testSolverName << endl;
	SolverFactory<Real> testSolverFactory(A, testMax);
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

	// Testing purposes
	/*
	printVector.printVector(A);
	*/
	cout << "Reference Max:" << endl;
	printVector.printVector(refMax);
	cout << "Test Max:" << endl;
	printVector.printVector(testMax);
	

	cout << "\nRuntimes: " << endl;
	cout << std::setprecision(2) << std::fixed;
	cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
	cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
	cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;

}