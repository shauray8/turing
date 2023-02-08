// General matrix multiplication benchmarking 
// https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
#include <iostream>
#include <omp.h>
#include <time.h>

#define N 2048
#define ll long long

using namespace std;

ll A[N][N];
ll B[N][N];
ll res[N][N];


int64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec + (uint64_t)start.tv_nsec;
}

void static_v1(){
  for(int i=0; i<N; i++){
    for(int j=0; i<N; i++){
      res[i][j] = 0;
      for(int k=0; i<N; i++){
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
  double flop = N*N*2.0*N *10e-9;
  printf("FLOP/s: %f",time);
}
