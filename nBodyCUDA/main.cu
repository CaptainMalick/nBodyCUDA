#include <iostream>
#include "constants.cuh"
#include <string>
#include "particle.cuh"
#include "SFML/Graphics.hpp"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "utils.h"
#include <cuda.h>

using namespace std;

// Number of particles, set once by user.
static unsigned long numParticles;
static particle* particleArr;
static particle* d_particleArr;

const int NUM_BLOCKS = 512;
static int BLOCK_SIZE;



// Creates initial conditions in terms of particle positions and velocities
//In this case: Annulus .05 AU thick, .05 to 1.5 AU radius, perfecty orbits, randomly distributed
void testCase() {
	srand(time(nullptr));
	size_t numAccept = 0;
	while (numAccept < numParticles) {
		double x = 3 * AU * (0.5 - doubleRand());
		double y = 3 * AU * (0.5 - doubleRand());
		double z = .05 * AU * (0.5 - doubleRand());
		if (pow(x, 2) + pow(y, 2) > pow(1.5 * AU, 2) || pow(x, 2) + pow(y, 2) < pow(.5 * AU, 2))
			continue;
		double posArr[3] = { x, y, z };
		double r = pow(pow(x, 2) + pow(y, 2), .5);
		double factor = pow(G * SM / r, .5);
		double xvel = -y * factor / r;
		double yvel = x * factor / r;
		double velArr[3] = { xvel, yvel, 0 };
		particleArr[numAccept] = particle(PM, posArr, velArr);
		numAccept++;
	}
}


bool initSimulation() {
	cout << "Enter number of particles to simulate" << endl;
	string numParticlesStr;
	cin >> numParticlesStr;
	numParticles = stod(numParticlesStr);
	PM = TPM / numParticles; //Based on constants, calculated mass of particles
	BLOCK_SIZE = (numParticles + NUM_BLOCKS - 1) / NUM_BLOCKS;

	particleArr = (particle *) malloc(numParticles * sizeof(particle));
	if (particleArr == NULL)
	{
		cout << "Heap allocation failed." << endl;
		return false;
	}

	testCase();
	if (cudaMalloc(&d_particleArr, numParticles * sizeof(particle)) != cudaSuccess)
	{
		free(particleArr);
		cout << "GPU memory allocation failed." << endl;
		return false;
	}

	if (cudaMemcpy(d_particleArr, particleArr, numParticles * sizeof(particle),
		cudaMemcpyHostToDevice) != cudaSuccess)
	{
		free(particleArr);
		cudaFree(d_particleArr);
		cout << "Memory copy failed." << endl;
		return false;
	}

	return true;
}



__global__
void d_kickDriftAll(particle* d_particleArr, unsigned long numParticles)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numParticles)
		d_particleArr[id].kickDrift();
}

__global__
void d_kickAll(particle* d_particleArr, unsigned long numParticles)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numParticles)
		d_particleArr[id].kick();
}

__global__
void d_updateAccAll(particle* d_particleArr, unsigned long numParticles)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numParticles) 
	{
		particle* part = &d_particleArr[id];
		// External field 
		part->setAcc(-d_G * d_SM * part->getPos().getUnit() / part->getPos().magSq());
		for (unsigned j = 0; j < numParticles; j++)
		{
			if (id != j)
			{
				mvec rVec = part->getPos() - d_particleArr[j].getPos();
				mvec rHat = rVec.getUnit();
				double factor = -d_G / (rVec.magSq() + pow(SOFTENING, 2.));
				part->setAcc(part->getAcc() + part->getMass() * factor * rHat);
			}
		}
	}
}

__global__
void d_doAll(particle* d_particleArr, unsigned long numParticles)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numParticles)
	{
		particle* part = &d_particleArr[id];
		part->kickDrift();
		// External field 
		part->setAcc(-d_G * d_SM * part->getPos().getUnit() / part->getPos().magSq());
		for (unsigned j = 0; j < numParticles; j++)
		{
			if (id != j)
			{
				mvec rVec = part->getPos() - d_particleArr[j].getPos();
				mvec rHat = rVec.getUnit();
				double factor = -d_G / (rVec.magSq() + pow(SOFTENING, 2.));
				part->setAcc(part->getAcc() + part->getMass() * factor * rHat);
			}
		}
		part->kick();
	}
}

void kickDriftAll(particle* d_particleArr, unsigned long numParticles)
{
	d_kickDriftAll<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_particleArr, numParticles);
}

void kickAll(particle* d_particleArr, unsigned long numParticles)
{
	d_kickAll<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_particleArr, numParticles);
}

void updateAccAll(particle* d_particleArr, unsigned long numParticles)
{
	d_updateAccAll<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_particleArr, numParticles);
}

double simulateStep() {
	/*kickDriftAll(d_particleArr, numParticles);
	updateAccAll(d_particleArr, numParticles);
	kickAll(d_particleArr, numParticles);*/
	d_doAll<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_particleArr, numParticles);
	if (cudaMemcpy(particleArr, d_particleArr, numParticles * sizeof(particle),
		cudaMemcpyDeviceToHost) != cudaSuccess)
		cout << "Failed to copy memory to host" << endl;
	return 0; 
}


int main() {
	if (!initSimulation()) {
		cout << "Could not initialize simulation. Exiting..." << endl;
		return 0;
	}


	simulateStep();
	int pixels_2 = 1000;
	int scale = 3;
	sf::RenderWindow window(sf::VideoMode(2000, 2000), "nBodyCUDA");
	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}
		window.clear(sf::Color(0, 0, 0));
		for (unsigned i = 0; i < numParticles; i++) {
			double mass = particleArr[i].getMass();
			double wtf = mass / TPM;
			float rad = 20 * std::pow(wtf, .3333);
			double xpos = particleArr[i].getPos()[0];
			double ypos = particleArr[i].getPos()[1];
			xpos = pixels_2 * ((xpos / scale) / AU) + pixels_2;
			ypos = pixels_2 * ((ypos / scale) / AU) + pixels_2;
			sf::CircleShape circ(rad, 5);
			circ.setPosition(xpos, ypos);
			circ.setFillColor(sf::Color(100, 250, 50));
			window.draw(circ);
		}
		window.display();
		simulateStep();
	}

	return 0;
}