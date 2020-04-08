#pragma once

class particleBase
{
public:
	particleBase() {}

	~particleBase() {}

	virtual bool init() = 0;

	virtual bool integrate() = 0;

	virtual bool display() = 0;

	virtual void cleanup() {}

	virtual void printStats() {}

};