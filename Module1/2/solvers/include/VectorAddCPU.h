#pragma once

#include "../interface/IVectorAdd.h"

#include <vector>

template <typename T>
class VectorAddCPU : public IVectorAdd<T>
{
private:
    
public:
    VectorAddCPU(const std::vector<T>& a,
        const std::vector<T>& b,
        std::vector<T>& c) : IVectorAdd<T>(a, b, c) {}
    
    virtual ~VectorAddCPU() {}

    void vectorAdd() override;
};
