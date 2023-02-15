// python3 gemm.py && g++ -funroll-loops -fopenmp gemm.cpp -o gemm -O3 -ffast-math -march=core2 && ./gemm
// General matrix multiplication benchmarking 
// https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
#include <iostream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <math.h>
#include <immintrin.h>

#define N 1024
#define ll long long
#define NUM_WORKERS 8
#define BLOCK 8

using namespace std;

// slows the performance down !
//vector<vector<float>> A(N, vector<float>(N,1));
//vector<vector<float>> B(N, vector<float>(N,1));
//vector<vector<float>> res(N, vector<float>(N,0));

float A[N][N];
float B[N][N];
float res[N][N];
float val[N][N];

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *resm = (__m256*)res;

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
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      if (fabsf(res[i][j]-val[i][j]) > 1e-3){
        printf("MISMATCH AT %d, %d :: %f != %f",i,j,res[i][j], val[i][j]);
      }
    }
  }
}

// SIMD on this 
// first try without the SIMD but with 8 variable's simul !1
void static_v2(){
  // i goes from 0-8, 9-16, 17-24 that is 8 after 8
  for(int i=0; i<N; i+=BLOCK){
    // j goes from 0-8, 9-16, 17-24 that is 8 after 8
    for(int j=0; j<N; j+=BLOCK){
      // till here we have i : 0-8 and j : 0-8
      float tc[BLOCK];
      for(int k=0; k<BLOCK; k++){
        // here we have an array with A : 0-8 * B : 0-8 in 0-8 slots
        // make this __m256
        // there is something called multiply and add together ! (SIMD)
         tc[k] = A[i][k] * B[k][j];
      }
      // add sum of tc to C[i][j]
      //C[i][j] = sum(tc);
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

// using __m256 to optmize strassen !
void strassen_v2(){
  for(int ii=0; ii<N; ii+=BLOCK){
    for(int jj=0; jj<N; jj+=BLOCK){

      __m256 temp[BLOCK];
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

// the compiler does use some XMM's but not very efficiently and no YMM's at all (?? do I not have AVX 2 ??)
// my processor does support AVX2 its just the compiler ! making it better 
int main(){

  FILE *f = fopen("./tmp/data","rb");
  fread(A, 1, sizeof(float)*N*N, f);
  fread(B, 1, sizeof(float)*N*N, f);
  fread(val, 1, sizeof(float)*N*N, f);
  fclose(f);

  uint64_t start = nanos();
  dynamic_v1();
  uint64_t end = nanos();
  double time = double(end-start)*1e-9;
  double flop = (N*N*2.0*N) *1e-9;
  printf("GFLOP/s: %f\n",flop/time);
}

