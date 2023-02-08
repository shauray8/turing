// General matrix multiplication benchmarking 
// https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
#include <iostream>
#include <omp.h>
#include <time.h>

#define N 1024
#define ll long long

using namespace std;

float A[N][N];
float B[N][N];
float res[N][N];


int64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

// runs on a single thread (pathetic performance !! gives off: 0.177351 GFLOP/s)
void static_v1(){
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      res[i][j] = 0;
      for(int k=0; k<N; k++){
        res[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void dynamic_v1(){
  ;
}

int main(){
  uint64_t start = nanos();
  static_v1();
  uint64_t end = nanos();
  double time = double(end-start)*1e-9;
  double flop = (N*N*2.0*N) *1e-9;
  printf("GFLOP/s: %f\n",flop/time);
}
