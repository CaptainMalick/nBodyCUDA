#include <iostream>
#include "constants.cuh"
#include "particle.cuh"

__host__ __device__
particle::particle(double mass, double posElems[3], double velElems[3]) {
    this->mass = mass;
    pos = mvec(posElems);
    vel = mvec(velElems);

    double zeros[3];
    tvel = mvec(zeros);
}

__host__ __device__
const mvec& particle::getPos() const {
    return pos;
}

__host__ __device__
const mvec& particle::getVel() const {
    return vel;
}

__host__ __device__
const mvec& particle::getTVel() const {
    return tvel;
}

__host__ __device__
const mvec& particle::getAcc() const {
    return acc;
}

__host__ __device__
void particle::setPos(const mvec& newPos) {
    pos = newPos;
}

__host__ __device__
void particle::setVel(const mvec& newVel) {
    vel = newVel;
}

__host__ __device__
void particle::setTVel(const mvec& newTVel) {
    tvel = newTVel;
}

__host__ __device__
void particle::setAcc(const mvec& newAcc) {
    acc = newAcc;
}

__host__ __device__
double particle::getMass() const {
    return mass;
}

__host__ __device__
void particle::setMass(double newMass) {
    mass = newMass;
}

__host__ __device__
void particle::kickDrift() {
    tvel = vel + dt * acc / 2;
    pos += dt * tvel;
    acc *= 0;
}

__host__ __device__
void particle::kick() {
    vel = tvel + dt * acc / 2;
}