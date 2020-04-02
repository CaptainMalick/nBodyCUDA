#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class mvec
{
public:
    __host__ __device__ mvec(const double elems[3]);
    __host__ __device__ mvec();

    __host__ __device__ double magSq() const;
    __host__ __device__ double mag() const;
    __host__ __device__ mvec getUnit() const;

    __host__ __device__ double& operator[](size_t index);
    __host__ __device__ const double& operator[](size_t index) const;

    
    __host__ __device__ bool operator==(const mvec& other);
    __host__ __device__ mvec& operator*=(double mult);
    __host__ __device__ mvec& operator/=(double divi);
    __host__ __device__ mvec& operator+=(const mvec& add);
    __host__ __device__ mvec& operator-=(const mvec& sub);
    __host__ __device__ mvec operator-();

    __host__ __device__ friend mvec operator*(mvec v, double mult);
    __host__ __device__ friend mvec operator*(double mult, mvec v);
    __host__ __device__ friend mvec operator/(mvec v, double divi);
    __host__ __device__ friend mvec operator+(mvec add1, const mvec& add2);
    __host__ __device__ friend mvec operator-(mvec sub1, const mvec& sub2);
    __host__ __device__ friend std::ostream& operator<<(std::ostream& os, const mvec& v);
private:
    double elements[3];
};
