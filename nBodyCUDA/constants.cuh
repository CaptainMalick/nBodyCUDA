#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Time step.
__constant__ const static float dt = 60 * 60 * 24 * 365 / 30000; // 29.2 hrs


// Astronomical unit.
const static float AU = 1.5e11;


/* Including -G in mass constants saves us previous multiplication operations. */

// Must be defined, used in __host__/__device__ functions
#define GRAVITATIONAL_CONSTANT 6.67408e-11f

/* Masses to a factor of neg_G to save on uncessary mul operations */
// Solar mass 
const static float SOLAR_MASS = 1.989e30;
__constant__ const static float D_SOLAR_MASS = 1.989e30;

// Total particle mass.
const static float TOTAL_PARTICLE_MASS = .001 * SOLAR_MASS;

// Softening factor.
__constant__ const static float SOFTENING = 1.5e9; //Used to remove singularities in force calculations
