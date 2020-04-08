#include "particle.cuh"
#include "mvec.cuh"

#define BLOCK_SIZE 512
#define SHARED_BUFFER_SIZE 48 * 1024


template <typename T>
__global__
void dk_kickDriftAll(struct posMass_s<T>* posMassArr, mvec<T>* velArr, mvec<T>* accArr, unsigned long NUM_PARTICLES)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < NUM_PARTICLES)
	{
		velArr[id] += (T)dt * accArr[id] * .5;
		posMassArr[id].pos += (T)dt * velArr[id];
		accArr[id] *= 0;
	}
}


template <typename T>
__global__
void dk_kickAll(mvec<T>* velArr, mvec<T>* accArr, unsigned long NUM_PARTICLES)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < NUM_PARTICLES)
		velArr[id] += (T)dt * accArr[id] * .5;
}

__device__
double inverseSquareRoot(double value)
{
	return rsqrt(value);
}

__device__
float inverseSquareRoot(float value)
{
	return rsqrtf(value);
}

template <typename T>
__global__
void dk_updateAccAll(mvec<T>* accArr, struct posMass_s<T>* posMassArr, unsigned long NUM_PARTICLES)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int sharedParticleLimit = SHARED_BUFFER_SIZE / sizeof(posMass_s<T>);
	__shared__ uint8_t sharedPosMassBuf[sharedParticleLimit * sizeof(struct posMass_s<T>)];
	struct posMass_s<T>* s_posMassArr = (struct posMass_s<T>*)& sharedPosMassBuf;
	

	if (id < NUM_PARTICLES)
	{
		int written = 0;
		mvec<T> currPos = posMassArr[id].pos;
		mvec<T> currAcc = accArr[id];

		/* Must be doule precision to prevent overflow MAG_EXT_P6. */
		double mag_ext_p6 = currPos.magSq();
		mag_ext_p6 *= mag_ext_p6 * mag_ext_p6;
		accArr[id] = -GRAVITATIONAL_CONSTANT * D_SOLAR_MASS * currPos / sqrt(mag_ext_p6);

		while (written < NUM_PARTICLES)
		{

			int write = NUM_PARTICLES - written;
			if (write > sharedParticleLimit)
				write = sharedParticleLimit;


			unsigned j;
#pragma unroll
			/* This is safe as the number of particles is rounded to BLOCK_SIZE. */
			for (j = 0; j < write / BLOCK_SIZE; j++)
				s_posMassArr[j * threadIdx.x] = posMassArr[written + j * threadIdx.x];

			__syncthreads();


#pragma unroll (32)
			for (j = 0; j < write; j++)
			{
				mvec<T> rVec = currPos - s_posMassArr[j].pos;
				T num = rVec.magSq() + SOFTENING * SOFTENING;
				num *= num * num;
				num = inverseSquareRoot(num);

				/* Storing mass up to negative G saves a multiplication and turns
				   a subtraction into an addition, allowing for an FMA instruction to be used.
				   Results in ~10% performance improvement. */

				currAcc += num * s_posMassArr[j].negGMass * rVec;
			}
			written += write;
		}

		accArr[id] = currAcc;
	}
}

template <typename T>
void particle<T>::d_kickDriftAll()
{
	dk_kickDriftAll<T> << <numBlocks, BLOCK_SIZE >> > (d_posMassArr, d_velArr, d_accArr, numParticles);
}

template <typename T>
void particle<T>::d_kickAll()
{
	dk_kickAll<T><< <numBlocks, BLOCK_SIZE >> > (d_velArr, d_accArr, numParticles);
}

template <typename T>
void particle<T>::d_updateAccAll()
{
	dk_updateAccAll <T><< <numBlocks, BLOCK_SIZE >> > (d_accArr, d_posMassArr, numParticles);
}

template class particle<float>;
template class particle<double>;