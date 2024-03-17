#pragma once

#include "../interface/IHello.h"
#include "../../utilities/include/GpuCommon.h"

#include <stdio.h>

class GpuHello : public IHello
{

public:
	virtual ~GpuHello() {}
	void hello() override;
};