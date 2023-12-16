// Solver interface 
#pragma once

#include "../../Parameters.h"

#include <vector>
#include <iostream>

template <typename T>
class IStencil
{
protected:
    const std::vector<T>& in;
    std::vector<T>& out;
        
public:
    IStencil(const std::vector<T>& in,
    std::vector<T>& out) : in{ in }, out{ out } {}
    virtual ~IStencil() {}
    virtual void stencil() = 0;
};

