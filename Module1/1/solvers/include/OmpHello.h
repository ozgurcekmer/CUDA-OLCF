#pragma once

#include "../interface/IHello.h"

#include <omp.h>

class OmpHello : public IHello
{

public:
	virtual ~OmpHello() {}
	void hello() override;
};