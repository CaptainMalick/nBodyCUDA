#pragma once
#include "mvec.cuh"
#include "constants.cuh"
#include "particle-base.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "SFML/Graphics.hpp"
#include <chrono>

template <typename T>
struct posMass_s {
    mvec<T> pos;
    T negGMass;
};


static double COPY_TIME = 0;
static double KERNEL_TIME = 0;

template <typename T>
class particle : public particleBase
{
public:

    particle(int numParticles, int numIterations);
    
    ~particle();

    bool init();

    bool integrate();

    bool display();

    void cleanup();

    void printStats();

private:
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;

    bool displayBool;
    
    sf::RenderWindow window;

    int counter;
    int iterations;
    struct posMass_s<T>* h_posMassArr;
    struct posMass_s<T>* d_posMassArr;
    mvec<T>* d_accArr;
    mvec<T>* d_velArr;
    

    int numParticles;
    int numBlocks;

    void testCase();

// Kernels in separate file
    void d_kickDriftAll();

    void d_kickAll();
    
    void d_updateAccAll();
};