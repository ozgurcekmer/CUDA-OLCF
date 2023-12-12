#pragma once

#include "../interface/IHello.h"

class SerialHello : public IHello
{

public:
	virtual ~SerialHello() {}
	void hello() override;
};