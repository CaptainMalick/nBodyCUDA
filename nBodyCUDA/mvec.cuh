#pragma once
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class mvec
{
public:
    __host__ __device__
    mvec(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    __host__ __device__
    mvec() {
        this->x = 0;
        this->y = 0;
        this->z = 0;
    }

    __host__ __device__
     float getX() const {
        return this->x;
    }

    __host__ __device__
        float getY() const {
        return this->y;
    }

    __host__ __device__
        float getZ() const {
        return this->z;
    }

    __host__ __device__
    float magSq() const {
        return this->x * this->x
            +  this->y * this->y 
            +  this->z * this->z;
    }

    __host__ __device__
    float mag() const {
        return sqrt(this->magSq());
    }

    __host__ __device__
    mvec getUnit() const {
        float mag = this->mag();
        return *this / mag;
    }

    __host__ __device__
    bool operator==(const mvec& other) {
        return this->x == other.x
            && this->y == other.y
            && this->z == other.z;
    }

    __host__ __device__
    mvec& operator*=(float mult) {
        this->x *= mult;
        this->y *= mult;
        this->z *= mult;

        return *this;
    }

    __host__ __device__
    mvec& operator/=(float divi) {
        this->x *= (1.0f / divi);
        this->y *= (1.0f / divi);
        this->z *= (1.0f / divi);

        return *this;
    }

    __host__ __device__
    mvec& operator+=(const mvec& add) {
        this->x += add.x;
        this->y += add.y;
        this->z += add.z;

        return *this;
    }

    __host__ __device__
    mvec& operator-=(const mvec& sub) {
        this->x -= sub.x;
        this->y -= sub.y;
        this->z -= sub.z;

        return *this;
    }

    __host__ __device__
    mvec operator-() {
        return -1 * *this;
    }

    __host__ __device__
    friend mvec operator*(mvec v, float mult) {
        v *= mult;
        return v;
    }

    __host__ __device__
    friend mvec operator*(float mult, mvec v) {
        v *= mult;
        return v;
    }
    __host__ __device__
    friend mvec operator/(mvec v, float divi) {
        v /= divi;
        return v;
    }

    __host__ __device__
    friend mvec operator+(mvec add1, const mvec& add2) {
        add1 += add2;
        return add1;
    }

    __host__ __device__
    friend mvec operator-(mvec sub1, const mvec& sub2) {
        sub1 -= sub2;
        return sub1;
    }

    friend std::ostream& operator<<(std::ostream& os, const mvec& v) {
        os << '[' << v.x << ", "
            << v.y << ", "
            << v.z << ']';

        return os;
    }

private:
    float x, y, z;
};
