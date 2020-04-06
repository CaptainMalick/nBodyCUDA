#pragma once
#include "mvec.cuh"
#include "constants.cuh"

class particle
{
public:
    __host__ __device__
    particle(float mass, float x, float y, float z, float vx, float vy, float vz) {
        this->setMass(mass);
        this->pos = mvec(x, y, z);
        this->vel = mvec(vx, vy, vz);
        this->tvel = mvec(0, 0, 0);
        this->acc = mvec(0, 0, 0);
    }

    __host__ __device__
    const mvec& getPos() const {
        return pos;
    }

    __host__ __device__
    const mvec& getVel() const {
        return vel;
    }

    __host__ __device__
    const mvec& getTVel() const {
        return tvel;
    }

    __host__ __device__
    const mvec& getAcc() {
        return acc;
    }

    __host__ __device__
    void setPos(const mvec& newPos) {
        pos = newPos;
    }

    __host__ __device__
    void setVel(const mvec& newVel) {
        vel = newVel;
    }

    __host__ __device__
    void setTVel(const mvec& newTVel) {
        tvel = newTVel;
    }

    __host__ __device__
    void setAcc(const mvec& newAcc) {
        acc = newAcc;
    }

    __host__ __device__
    float getMass() const {
        return -negGMass / GRAVITATIONAL_CONSTANT;
    }

    __host__ __device__
    float getNegGMass() const {
        return negGMass;
    }

    __host__ __device__
    void setMass(float newMass) {
        negGMass = -GRAVITATIONAL_CONSTANT * newMass;
    }

    __host__ __device__
    void kickDrift() {
        tvel = vel + dt * acc * (1. / 2);
        pos += dt * tvel;
        acc *= 0;
    }

    __host__ __device__
    void kick() {
        vel = tvel + dt * acc * (1. / 2);
    }

private:
    float negGMass;
    mvec pos;
    mvec vel;
    mvec tvel;
    mvec acc;
};