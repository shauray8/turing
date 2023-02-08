// General matrix multiplication benchmarking 
// https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
#include <iostream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <immintrin.h>

#define N 2048
#define ll long long
#define NUM_WORKERS 8
#define BLOCK 64

using namespace std;

// slows the performance down !
//vector<vector<float>> A(N, vector<float>(N,0));
//vector<vector<float>> B(N, vector<float>(N,0));
//vector<vector<float>> res(N, vector<float>(N,0));


float A[N][N];
float B[N][N];
float res[N][N];
__m256 *Am = (__m256*)A;

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

// g++ -fopenmp gemm.cpp -o gemm && ./gemm
// I don't actually know if it's right or not but it runs on all the threads and gives off: 1.350512
// this is probably wrong !
void dynamic_v1(){
  float acc = 0;
  #pragma omp parallel for schedule(static,NUM_WORKERS) reduction(+:acc)
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      acc = 0;
      for(int k=0; k<N; k++){
        acc += A[i][k] * B[j][k];
      }
      res[i][j] = acc;
    }
  }
}

// spliting into blocks of size BLOCK and then computing the matrix product (gives off no significant improvement)
void strassen(){
  for(int ii=0; ii<N; ii+=BLOCK){
    for(int jj=0; jj<N; jj+=BLOCK){

      for(int i=0; i<BLOCK; i++){
        for(int j=0; j<BLOCK; j++){
          for(int k=0; k<N; k++){
            res[ii+i][jj+j] += A[ii+i][k] * B[jj+j][k];
          }
        }
      }

    }
  }
}

int main(){
  uint64_t start = nanos();
  dynamic_v1();
  uint64_t end = nanos();
  double time = double(end-start)*1e-9;
  double flop = (N*N*2.0*N) *1e-9;
  printf("GFLOP/s: %f\n",flop/time);
}

