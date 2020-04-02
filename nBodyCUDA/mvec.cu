#include "mvec.cuh"

#include <cmath>
#include <sstream>
using namespace std;

ostream& operator<<(ostream& os, const mvec& v);

__host__ __device__
mvec::mvec(const double elems[3]) {
    elements[0] = elems[0];
    elements[1] = elems[1];
    elements[2] = elems[2];
}

__host__ __device__
mvec::mvec() {
    elements[0] = 0;
    elements[1] = 0;
    elements[2] = 0;
}


__host__ __device__
double mvec::magSq() const {
    double sum = 0;
    for (double e : elements) {
        sum += pow(e, 2);
    }
    return sum;
}

__host__ __device__
double mvec::mag() const {
    return sqrt(this->magSq());
}

__host__ __device__
mvec mvec::getUnit() const {
    double mag = this->mag();
    return *this / mag;
}

__host__ __device__
double& mvec::operator[](size_t index) {
    return elements[index];
}

__host__ __device__
const double& mvec::operator[](size_t index) const {
    return elements[index];
}

__host__ __device__
bool mvec::operator==(const mvec& other) {
    for (size_t i = 0; i < 3; i++)
        if (elements[i] != other.elements[i])
            return false;
    return true;
}

__host__ __device__
mvec& mvec::operator*=(double mult) {
    elements[0] *= mult;
    elements[1] *= mult;
    elements[2] *= mult;
    
    return *this;
}

__host__ __device__
mvec& mvec::operator/=(double divi) {
    elements[0] /= divi;
    elements[1] /= divi;
    elements[2] /= divi;
    
    return *this;
}

__host__ __device__
mvec& mvec::operator+=(const mvec& add) {
    elements[0] += add[0];
    elements[1] += add[1];
    elements[2] += add[2];
    
    return *this;
}

__host__ __device__
mvec& mvec::operator-=(const mvec& sub) {
    elements[0] -= sub[0];
    elements[1] -= sub[1];
    elements[2] -= sub[2];

    return *this;
}

__host__ __device__
mvec mvec::operator-() {
    return -1 * *this;
}

__host__ __device__
mvec operator*(mvec v, double mult) {
    v *= mult;
    return v;
}

__host__ __device__
mvec operator*(double mult, mvec v) {
    v *= mult;
    return v;
}

__host__ __device__
mvec operator/(mvec v, double divi) {
    v /= divi;
    return v;
}

__host__ __device__
mvec operator+(mvec add1, const mvec& add2) {
    add1 += add2;
    return add1;
}

__host__ __device__
mvec operator-(mvec sub1, const mvec& sub2) {
    sub1 -= sub2;
    return sub1;
}

ostream& operator<<(ostream& os, const mvec& v) {
    os << '[' << v.elements[0] << ", " 
              << v.elements[1] << ", " 
              << v.elements[2] << ']';
    
    return os;
}
