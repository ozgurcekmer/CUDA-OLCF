// Solver interface 
#pragma once

#include "../../Parameters.h"

#include <iostream>

class IHello
{
protected:
    
public:
    virtual ~IHello() {}

    // Public methods
    virtual void hello() = 0;
    
};