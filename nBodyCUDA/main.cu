#include <iostream>
#include "constants.cuh"
#include <string>
#include "particle.cuh"
#include "SFML/Graphics.hpp"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "utils.h"
#include <chrono>
#include <cuda.h>

using namespace std;


int main() {
	while (true)
	{
		std::cout << "Enter number of particles to simulate (will round up to nearest 256: " << std::endl;
		std::string numParticleStr;
		cin >> numParticleStr;
		int numParticles = stoi(numParticleStr);

		std::cout << "Enter number of iterations (0 for no limit): " << std::endl;
		std::string numIterationStr;
		cin >> numIterationStr;
		int numIterations = stoi(numIterationStr);

		std::cout << "Enter precision (1 for single, 2 for double)" << std::endl;
		std::string precisionStr;
		cin >> precisionStr;
		int precision = stoi(precisionStr);

		particleBase* particleSystem = NULL;
		if (precision == 1)
			particleSystem = new particle<float>(numParticles, numIterations);
		else if (precision == 2)
			particleSystem = new particle<double>(numParticles, numIterations);
		else return 0;

		particleSystem->init();
		while (true)
		{
			if (!particleSystem->integrate() ||
				!particleSystem->display())
				break;

		}

		particleSystem->cleanup();
		particleSystem->printStats();
		delete particleSystem;

		cout << "Enter Q to exit..." << endl;
		string exitStr;
		cin >> exitStr;
		if (exitStr == "Q")
			break;
	}

	return 0;
}