// Solver interface 
#pragma once

#include "../../Parameters.h"

#include <vector>
#include <iostream>

template <typename T>
class IVectorAdd
{
protected:
    const std::vector<T>& a;
    const std::vector<T>& b;
    std::vector<T>& c;
        
public:
    IVectorAdd(const std::vector<T>& a,
    const std::vector<T>& b,
    std::vector<T>& c) : a{ a }, b{ b }, c{ c } {}
    virtual ~IVectorAdd() {}
    virtual void vectorAdd() = 0;
};

