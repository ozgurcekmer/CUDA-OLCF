// Solver interface 
#pragma once

#include "../../Parameters.h"

#include <vector>
#include <iostream>

template <typename T>
class IMatrixMult
{
protected:
    const std::vector<T>& a;
    const std::vector<T>& b;
    std::vector<T>& c;
        
public:
    IMatrixMult(const std::vector<T>& a,
    const std::vector<T>& b,
    std::vector<T>& c) : a{ a }, b{ b }, c{ c } {}
    virtual ~IMatrixMult() {}
    virtual void matrixMult() = 0;
};

