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

// Number of particles, set once by user.
// static unsigned long NUM_PARTICLES_MAX;
static unsigned long NUM_PARTICLES;
static particle* particleArr;
static particle* d_particleArr;

static int NUM_BLOCKS;
static int BLOCK_SIZE = 256;
static float COPY_TIME = 0;
static float KERNEL_TIME = 0;

#define SHARED_PARTICLE_LIMIT 256

// Creates initial conditions in terms of particle positions and velocities
//In this case: Annulus .05 AU thick, .05 to 1.5 AU radius, perfecty orbits, randomly distributed
void testCase() {
	srand(time(nullptr));
	size_t numAccept = 0;
	while (numAccept < NUM_PARTICLES) {
		float x = 3 * AU * (0.5 - doubleRand());
		float y = 3 * AU * (0.5 - doubleRand());
		float z = .05 * AU * (0.5 - doubleRand());
		if (pow(x, 2) + pow(y, 2) > pow(1.5 * AU, 2) || pow(x, 2) + pow(y, 2) < pow(.5 * AU, 2))
			continue;
		float posArr[3] = { x, y, z };
		float r = sqrt(pow(x, 2) + pow(y, 2));
		float factor = sqrt(GRAVITATIONAL_CONSTANT * SOLAR_MASS / r);
		float vx = -y * factor / r;
		float vy = x * factor / r;

		particleArr[numAccept] = particle(PARTICLE_MASS, x, y, z, vx, vy, 0);
		numAccept++;
	}
}

void cleanupSimulation() {
	cudaFreeHost(particleArr);
	cudaFree(d_particleArr);
}

bool initSimulation() {
	std::cout << "Enter number of particles to simulate" << endl;
	string numParticlesStr;

	std::cin >> numParticlesStr;
	
	NUM_PARTICLES = stod(numParticlesStr);
	NUM_PARTICLES = (NUM_PARTICLES + BLOCK_SIZE - 1) - (NUM_PARTICLES + BLOCK_SIZE - 1) % BLOCK_SIZE;
	cout << "Particle count (rounded up): " << NUM_PARTICLES << endl;
	PARTICLE_MASS = TOTAL_PARTICLE_MASS / NUM_PARTICLES;
	NUM_BLOCKS = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaMallocHost(&particleArr, NUM_PARTICLES * sizeof(particle));
	if (particleArr == NULL)
	{
		std::cout << "Heap allocation failed." << endl;
		return false;
	}

	testCase();
	if (cudaMalloc(&d_particleArr, NUM_PARTICLES * sizeof(particle)) != cudaSuccess)
	{
		cleanupSimulation();
		std::cout << "GPU memory allocation failed." << endl;
		return false;
	}

	if (cudaMemcpy(d_particleArr, particleArr, NUM_PARTICLES * sizeof(particle),
		cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cleanupSimulation();
		std::cout << "Memory copy failed." << endl;
		return false;
	}

	return true;
}



__global__
void d_kickDriftAll(particle* d_particleArr, unsigned long NUM_PARTICLES)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < NUM_PARTICLES)
		d_particleArr[id].kickDrift();
}

__global__
void d_kickAll(particle* d_particleArr, unsigned long NUM_PARTICLES)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < NUM_PARTICLES)
		d_particleArr[id].kick();
}


__global__
void d_updateAccAll(particle* d_particleArr, unsigned long NUM_PARTICLES)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ uint8_t sharedParticleBuf[SHARED_PARTICLE_LIMIT * sizeof(particle)];
	particle* sharedParticleArr = (particle*)&sharedParticleBuf;


	if (id < NUM_PARTICLES)
	{
		int written = 0;
		particle* part = &d_particleArr[id];
		
		// Must be doule precision to prevent overflow MAG_EXT_P6
		double mag_ext_p6 = part->getPos().magSq();
		mag_ext_p6 *= mag_ext_p6 * mag_ext_p6;
		part->setAcc(-GRAVITATIONAL_CONSTANT * D_SOLAR_MASS * part->getPos() / sqrt(mag_ext_p6)); 

		while (written < NUM_PARTICLES)
		{

			int write = NUM_PARTICLES - written;
			if (write > SHARED_PARTICLE_LIMIT)
				write = SHARED_PARTICLE_LIMIT;
			sharedParticleArr[threadIdx.x] = d_particleArr[written + threadIdx.x];
	
			__syncthreads();


#pragma unroll (32)
			for (unsigned j = 0; j < write; j++)
			{
				const particle* partOther = &sharedParticleArr[j];
				mvec rVec = part->getPos() - partOther->getPos();
				float num = rVec.magSq() + SOFTENING * SOFTENING;
				num *= num * num;
				num = rsqrtf(num);

				/* Storing mass up to negative G saves a multiplication and turns
				   a subtraction into an addition, allowing for an FMA instruction to be used.
				   Results in ~10% performance improvement. */
				part->setAcc(part->getAcc() + num * partOther->getNegGMass() * rVec);
			}
			written += write;
		}
	}
}


void kickDriftAll(particle* d_particleArr, unsigned long NUM_PARTICLES)
{
	d_kickDriftAll<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_particleArr, NUM_PARTICLES);
}

void kickAll(particle* d_particleArr, unsigned long NUM_PARTICLES)
{
	d_kickAll<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_particleArr, NUM_PARTICLES);
}


void updateAccAll(particle* d_particleArr, unsigned long NUM_PARTICLES)
{
	d_updateAccAll << <NUM_BLOCKS, BLOCK_SIZE >> > (d_particleArr, NUM_PARTICLES);	
}

bool simulateStep()
{
		auto t1 = chrono::high_resolution_clock::now();
		kickDriftAll(d_particleArr, NUM_PARTICLES);
		updateAccAll(d_particleArr, NUM_PARTICLES);
		kickAll(d_particleArr, NUM_PARTICLES);
		cudaDeviceSynchronize();
		auto t2 = chrono::high_resolution_clock::now();

		KERNEL_TIME += chrono::duration<float>(t2 - t1).count();;

		if (cudaMemcpy(particleArr, d_particleArr, NUM_PARTICLES * sizeof(particle),
			cudaMemcpyDeviceToHost) != cudaSuccess)
			std::cout << "Failed to copy memory to host" << endl;
		auto t3 = chrono::high_resolution_clock::now();
		COPY_TIME += chrono::duration<float>(t3 - t2).count();

		return true;
}


int main() {
	bool BENCHMARK = false;
	if (!initSimulation()) {
		std::cout << "Could not initialize simulation. Exiting..." << endl;
		return 0;
	}

	std::cout << "PARTICLE SIZE: " << sizeof(particle) << endl;
	sf::RenderWindow window(sf::VideoMode(2000, 2000), "nBodyCUDA");
	int pixels_2 = 1000;
	int scale = 3;
	int j = 1;
	auto t1 = chrono::high_resolution_clock::now();
	simulateStep();
	
	while (window.isOpen())
	{
		if (!BENCHMARK)
		{
			sf::Event event;
			if (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
					window.close();
			}
			window.clear(sf::Color(0, 0, 0));

			for (unsigned i = 0; i < NUM_PARTICLES; i++) {
				/*float mass = particleArr[i].getMass();
				float wtf = mass / TOTAL_PARTICLE_MASS;
				float rad = 20 * std::pow(wtf, .3333);
				float xpos = particleArr[i].getPos().getX();
				float ypos = particleArr[i].getPos().getY();
				xpos = pixels_2 * ((xpos / scale) / AU) + pixels_2;
				ypos = pixels_2 * ((ypos / scale) / AU) + pixels_2;*/
				sf::CircleShape circ(6, 5);
				circ.setPosition(0, 0);
				circ.setFillColor(sf::Color(100, 250, 50));
				window.draw(circ);
			}
			window.display();
		}
		if (!simulateStep())
			break;
		if (j++ % 100 == 0) std::cout << j << endl;
		//if (j == 1000) break;
	}
	auto t2 = chrono::high_resolution_clock::now();
	float duration = chrono::duration<float>(t2 - t1).count();

	window.close();

	std::cout << "Number of frames: " << j << endl;
	std::cout << "Interactions per second (Billions): " << (j * NUM_PARTICLES * (float)NUM_PARTICLES / 1e9) / duration << endl << endl;

	std::cout << "KERNEL TIME (sec): " << KERNEL_TIME << endl;
	std::cout << "KERNEL PERCENT: " << 100 * KERNEL_TIME / duration << endl << endl;
	std::cout << "COPY TIME (sec): " << COPY_TIME << endl;
	std::cout << "COPY PERCENT: " << 100 * COPY_TIME / duration << endl << endl;
	float unaccounted = duration - KERNEL_TIME - COPY_TIME;
	std::cout << "UNACCOUNTED TIME (sec): " << unaccounted << endl;
	std::cout << "UNACCOUNTED PERCENT: " << 100 * unaccounted / duration << endl << endl;
	std::cout << "TOTAL TIME: (sec)" << duration << endl;

	std::cout << endl << "FPS: " << j / duration << endl;

	std::cout << "Press Enter to exit.." << endl;

	string unused;
	std::cin >> unused;

	cleanupSimulation();

	return 0;
}