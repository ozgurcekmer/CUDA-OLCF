// Standard Libraries
#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>

// Parameters
#include "Parameters.h"

// Utilities
#include "utilities/include/MaxError.h"
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
    cout << "Nvidia Blog: How to Overlap Data Transfers in CUDA/C++" << endl;
    cout << "Vector Size: " << N << endl;

    // Maximum error evaluator
    MaxError<Real> maximumError;
    
    Vector::pinnedVector<Real> x(N, 0.0);
    Vector::pinnedVector<Real> yRef(N, 0.0);
    Vector::pinnedVector<Real> yTest(N, 0.0);
    
    RandomVectorGenerator<Real> randomVector;
    randomVector.randomVector(x);

    WarmupGPU warmupGPU;
    warmupGPU.setup(refGPU, testGPU);
   
    // Reference solver
    cout << "\nSolver: " << refSolverName << endl;
    SolverFactory<Real> refSolverFactory(x, yRef);
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

    // Test gridder
    cout << "\nSolver: " << testSolverName << endl;
    SolverFactory<Real> testSolverFactory(x, yTest);
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

    cout << "\nVerifying the test code" << endl;
    maximumError.maxError(yRef, yTest);

    cout << "\nRuntimes: " << endl;
    cout << std::setprecision(3) << std::fixed;
    cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
    cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
    cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;
}
