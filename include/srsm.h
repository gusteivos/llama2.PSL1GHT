
#ifndef SCALAR_RMSNORM_SOFTMAX_MATMUL_H
#define SCALAR_RMSNORM_SOFTMAX_MATMUL_H

#include <errno.h>

#include <math.h>



void scalar_rmsnorm(float* o, float* x, float* weight, int size);

void scalar_in_place_softmax(float* x, int size);

void scalar_matmul(float* xout, float* x, float* w, int n, int d);

#endif/*SCALAR_RMSNORM_SOFTMAX_MATMUL_H*/
