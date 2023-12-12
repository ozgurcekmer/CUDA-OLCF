#include "Parameters.h"

#include "solvers/factory/SolverFactory.h"

#include <memory>
#include <iostream>

using std::cout;
using std::endl;

using std::shared_ptr;
using std::make_unique;

int main()
{
	cout << "Hello world project" << endl;
	cout << "\n" << solverName << " is working." << endl;

	SolverFactory solverFactory;
	std::shared_ptr<IHello> solver = solverFactory.getSolver(solverName);
	solver->hello();
	

}
