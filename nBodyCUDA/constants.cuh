#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Time step.
__constant__ const static double dt = 60 * 60 * 24 * 365 / 300; // 29.2 hrs
// Gravitational constant.
const static double G = 6.67408 * 10e-11;
__constant__ const static double d_G = 6.67408 * 10e-11;
// Astronomical unit.
const static double AU = 1.5e11;
// Solar mass.
const static double SM = 1.989e30; //solar mass
__constant__ const static double d_SM = 1.989e30; //solar mass
// Total particle mass.
const static double TPM = .001 * SM;
// Particle mass.
static double PM;
// Softening factor.
__constant__ const static double SOFTENING = 1.5e9; //Used to remove singularities in force calculations
const static int MAX_DEPTH = 10;