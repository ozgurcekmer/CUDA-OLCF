#pragma once

#include "../interface/ISolver.h"

#include <vector>
#include <iostream>
#include <omp.h>

template <typename T>
class CpuSolver : public ISolver<T>
{
private:

public:
    CpuSolver(std::vector<T>& A, std::vector<T>& B) : ISolver<T>(A, B) {}

    virtual ~CpuSolver() {}
    void solver() override;
};
