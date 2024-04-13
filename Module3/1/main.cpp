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

// Solver factory
#include "solvers/factory/VectorAddFactory.h"

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
    
    vector<Real> a(N, 0.0);
    vector<Real> b(N, 0.0);
    vector<Real> cRef(N, 0.0);
    vector<Real> cTest(N, 0.0);
    
    randomVector.randomVector(a);
    randomVector.randomVector(b);

    WarmupGPU warmupGPU;
    warmupGPU.setup(refGPU, testGPU);

    cout << "RefGPU = " << refGPU << endl;
    cout << "TestGPU = " << testGPU << endl;
   
    // Reference solver
    cout << "\nSolver: " << refSolverName << endl;
    VectorAddFactory<Real> refSolverFactory(a, b, cRef);
    std::shared_ptr<IVectorAdd<Real>> refSolver = refSolverFactory.getSolver(refSolverName);
    if (refGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for reference solver: " << refSolverName << endl;
    }
    auto tInit = omp_get_wtime();
    refSolver->vectorAdd();
    auto tFin = omp_get_wtime();
    auto runtimeRef = (tFin - tInit) * 1000.0; // in ms

    if ((!refGPU) && testGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for test solver: " << testSolverName << endl;
    }

    // Test gridder
    cout << "\nSolver: " << testSolverName << endl;
    VectorAddFactory<Real> testSolverFactory(a, b, cTest);
    std::shared_ptr<IVectorAdd<Real>> testSolver = testSolverFactory.getSolver(testSolverName);
    if ((!refGPU) && testGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for test solver: " << testSolverName << endl;
    }
    tInit = omp_get_wtime();
    testSolver->vectorAdd();
    tFin = omp_get_wtime();
    auto runtimeTest = (tFin - tInit) * 1000.0; // in ms
    
    cout << "\nVerifying the code" << endl;
    maximumError.maxError(cRef, cTest);

    cout << "\nRuntimes: " << endl;
    cout << std::setprecision(2) << std::fixed;
    cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
    cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
    cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;
}
