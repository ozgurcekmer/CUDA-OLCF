#pragma once

#include "../interface/ISolver.h"

#include <vector>
#include <iostream>
#include <omp.h>

template <typename T>
class CpuSolver : public ISolver<T>
{
private:
    T gpdf(T val);
    
public:
    CpuSolver(Vector::pinnedVector<T>& x, Vector::pinnedVector<T>& y) : ISolver<T>(x, y) {}

    virtual ~CpuSolver() {}
    void solver() override;
};
