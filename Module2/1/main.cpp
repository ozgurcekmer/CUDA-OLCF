// Standard Libraries
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>

// Parameters
#include "Parameters.h"

// Utilities
#include "utilities/include/MaxError.h"
#include "utilities/include/RandomVectorGenerator.h"
#include "utilities/include/WarmupGPU.h"
#include "utilities/include/PrintVector.h"

// Solver Factory
#include "solvers/factory/StencilFactory.h"

using std::cout;
using std::endl;
using std::vector;
using std::left;
using std::setprecision;
using std::setw;
using std::fixed;

int main()
{
    // Maximum error evaluator
    MaxError<Real> maximumError;

    RandomVectorGenerator<Real> randomVector;
    // Print vector object
    PrintVector<Real> printVector;

    vector<Real> in(N + 2 * RADIUS);
	vector<Real> outRef(N + 2 * RADIUS, 0.0);
    vector<Real> outTest(N + 2 * RADIUS, 0.0);

    randomVector.randomVector(in);

    WarmupGPU warmupGPU;
    warmupGPU.setup(refGPU, testGPU);

    cout << "RefGPU = " << refGPU << endl;
    cout << "TestGPU = " << testGPU << endl;

    // Reference solver
    cout << "\nSolver: " << refSolverName << endl;
    StencilFactory<Real> refSolverFactory(in, outRef);
    std::shared_ptr<IStencil<Real>> refSolver = refSolverFactory.getSolver(refSolverName);
    if (refGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for reference solver: " << refSolverName << endl;
    }
    auto tInit = omp_get_wtime();
    refSolver->stencil();
    auto tFin = omp_get_wtime();
    auto runtimeRef = (tFin - tInit) * 1000.0; // in ms

    // Test gridder
    cout << "\nSolver: " << testSolverName << endl;
    StencilFactory<Real> testSolverFactory(in, outTest);
    std::shared_ptr<IStencil<Real>> testSolver = testSolverFactory.getSolver(testSolverName);
    if ((!refGPU) && testGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for test solver: " << testSolverName << endl;
    }
    tInit = omp_get_wtime();
    testSolver->stencil();
    tFin = omp_get_wtime();
    auto runtimeTest = (tFin - tInit) * 1000.0; // in ms
    
    cout << "\nVerifying the code" << endl;
    maximumError.maxError(outRef, outTest);


    cout << "\nRuntimes: " << endl;
    cout << std::setprecision(6) << std::fixed;
    cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
    cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
    cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;

   // printVector.printVector(outRef);
    //printVector.printVector(outTest);

}
