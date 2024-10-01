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
    std::vector<T>& A;
    std::vector<T>& B;
        
public:
    ISolver(std::vector<T>& A, std::vector<T>& B) : A{ A }, B { B } {}
    virtual ~ISolver() {}
    virtual void solver() = 0;
};

