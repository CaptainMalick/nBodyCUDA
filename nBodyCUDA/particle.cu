#include "particle.cuh"
#include "utils.h"
#include <string.h>
#include <time.h>

#define BLOCK_SIZE 256

template <typename T>
particle<T>::particle(int numParticles, int numIterations) 
{
	this->iterations = numIterations;
	this->numParticles = (numParticles + BLOCK_SIZE - 1) - 
				         (numParticles + BLOCK_SIZE - 1) % BLOCK_SIZE;
	numBlocks = numParticles / BLOCK_SIZE;
	h_posMassArr = NULL;
	d_posMassArr = NULL;
	d_accArr = NULL;
	d_velArr = NULL;
	counter = 0;
	displayBool = false;
}

template <typename T>
particle<T>::~particle()
{
	cleanup();
}

template <typename T>
bool particle<T>::init()
{
	
	if (cudaMallocHost(&h_posMassArr, numParticles * sizeof(struct posMass_s<T>)) != cudaSuccess)
	{
		cleanup();
		std::cout << "GPU memory allocation failed." << std::endl;
		return false;
	}

	if (cudaMalloc(&d_posMassArr, numParticles * sizeof(struct posMass_s<T>)) != cudaSuccess)
	{
		cleanup();
		std::cout << "GPU memory allocation failed." << std::endl;
		return false;
	}

	if (cudaMalloc(&d_accArr, numParticles * sizeof(mvec<T>)) != cudaSuccess)
	{
		cleanup();
		std::cout << "GPU memory allocation failed." << std::endl;
		return false;
	}

	if (cudaMalloc(&d_velArr, numParticles * sizeof(mvec<T>)) != cudaSuccess)
	{
		cleanup();
		std::cout << "GPU memory allocation failed." << std::endl;
		return false;
	}

	testCase();

	if (displayBool)
		window.create(sf::VideoMode(2000, 2000), "nBodyCUDA");

	startTime = std::chrono::high_resolution_clock::now();
	return true;
}

template <typename T>
bool particle<T>::integrate()
{
	d_kickDriftAll();
	d_updateAccAll();
	d_kickAll();

	if (cudaMemcpy(h_posMassArr, d_posMassArr, numParticles * sizeof(struct posMass_s<T>),
		cudaMemcpyDeviceToHost) != cudaSuccess)
		std::cout << "Failed to copy memory to host" << std::endl;
	return true;
}

template <typename T>
bool particle<T>::display()
{
	if (++counter % 100 == 0)
		std::cout << "Iteration " << counter << std::endl;

	if (counter == iterations)
		return false;

	if (!displayBool)
		return true;

	if (!window.isOpen())
		return false;

	sf::Event event;
	if (window.pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
			window.close();
	}

	window.clear(sf::Color(0, 0, 0));
	int pixels_2 = 1000;
	int scale = 3;
	for (unsigned i = 0; i < numParticles; i++) {
		double mass = -h_posMassArr[i].negGMass / GRAVITATIONAL_CONSTANT;
		double wtf = mass / TOTAL_PARTICLE_MASS;
		double rad = 20 * std::pow(wtf, .3333);
		double xpos = h_posMassArr[i].pos.getX();
		double ypos = h_posMassArr[i].pos.getY();
		xpos = pixels_2 * ((xpos / scale) / AU) + pixels_2;
		ypos = pixels_2 * ((ypos / scale) / AU) + pixels_2;
		sf::CircleShape circ(rad, 5);
		circ.setPosition(xpos, ypos);
		circ.setFillColor(sf::Color(100, 250, 50));
		window.draw(circ);
	}
	window.display();

	return true;
}

template <typename T>
void particle<T>::cleanup()
{
	endTime = std::chrono::high_resolution_clock::now();

	cudaFreeHost(h_posMassArr);
	cudaFree(d_posMassArr);
	cudaFree(d_accArr);
	cudaFree(d_velArr);
}

template <typename T>
void particle<T>::printStats()
{
	double duration = std::chrono::duration<double>(endTime - startTime).count();
	std::cout << "Iterations: " << counter << std::endl;
	std::cout << "FPS: " << counter / duration << std::endl;
	double BIPS = counter;
	BIPS *= numParticles;
	BIPS *= numParticles;
	BIPS /= 1e9;
	BIPS /= duration;
	std::cout << "BIPS: " << BIPS << std::endl;
	std::cout << "TFLOPS: " << 20 * BIPS << std::endl;
}

template <typename T>
void particle<T>::testCase() {
	srand(time(nullptr));
	size_t numAccept = 0;
	mvec<T> *h_velArr = (mvec<T> *)malloc(numParticles * sizeof mvec<T>);
	while (numAccept < numParticles) {
		double x = 3 * AU * (0.5 - doubleRand());
		double y = 3 * AU * (0.5 - doubleRand());
		double z = .05 * AU * (0.5 - doubleRand());
		if (pow(x, 2) + pow(y, 2) > pow(1.5f * AU, 2) || 
			pow(x, 2) + pow(y, 2) < pow(.5f * AU, 2))
			continue;
		double r = sqrt(pow(x, 2) + pow(y, 2));
		double factor = sqrt(GRAVITATIONAL_CONSTANT * SOLAR_MASS / r);
		double vx = -y * factor / r;
		double vy = x * factor / r;

		double mass = TOTAL_PARTICLE_MASS / numParticles;
		h_posMassArr[numAccept].negGMass = -GRAVITATIONAL_CONSTANT * mass;
		h_posMassArr[numAccept].pos = mvec<T>(x, y, z);
		h_velArr[numAccept] = mvec<T>(vx, vy, 0);
		numAccept++;
	}

	if (cudaMemcpy(d_posMassArr, h_posMassArr, numParticles * sizeof(struct posMass_s<T>),
		cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cleanup();
		std::cout << "Memory copy failed." << std::endl;
	}

	if (cudaMemcpy(d_velArr, h_velArr, numParticles * sizeof(mvec<T>),
		cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cleanup();
		std::cout << "Memory copy failed." << std::endl;
	}
	free(h_velArr);
}

template class particle<float>;
template class particle<double>;