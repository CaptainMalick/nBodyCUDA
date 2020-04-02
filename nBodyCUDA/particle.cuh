#pragma once
#include "mvec.cuh"

class particle
{
public:
    __host__ __device__
    particle(double mass, double posElems[3], double velElems[3]);

    __host__ __device__
    const mvec& getPos() const;
    __host__ __device__
    const mvec& getVel() const;
    __host__ __device__
    const mvec& getTVel() const;
    __host__ __device__
    const mvec& getAcc() const;
    __host__ __device__
    void setPos(const mvec& newPos);
    __host__ __device__
    void setVel(const mvec& newVel);
    __host__ __device__
    void setTVel(const mvec& newTvel);
    __host__ __device__
    void setAcc(const mvec& newAcc);
    __host__ __device__
    double getMass() const;
    __host__ __device__
    void setMass(double newMass);
    __host__ __device__
    void kickDrift(); //applies kick-drift step of leapfrog to particle
    __host__ __device__
    void kick(); //applies kick step of leapfrog to particle

private:
    double mass;
    mvec pos;
    mvec vel;
    mvec tvel;
    mvec acc;
};