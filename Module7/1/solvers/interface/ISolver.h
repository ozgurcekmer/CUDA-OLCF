// Solver interface 
#pragma once

#include "../../Parameters.h"
#include "../../utilities/include/vectors/PinnedVector.h"

#include <vector>
#include <iostream>

template <typename T>
class ISolver
{
protected:
    Vector::pinnedVector<T>& x;
    Vector::pinnedVector<T>& y;

    const T MEAN = 0.0;
    const T SIGMA = 1.0;
        
public:
    ISolver(Vector::pinnedVector<T>& x, Vector::pinnedVector<T>& y) : x{ x }, y{ y } {}
    virtual ~ISolver() {}
    virtual void solver() = 0;
};

