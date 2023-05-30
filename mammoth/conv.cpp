//204 GFLOPS
// python3 gemm.py && g++ -funroll-loops -fopenmp gemm.cpp -o gemm -O3 -ffast-math -march=core2 && ./gemm
// General matrix multiplication benchmarking 
// https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
#include <iostream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <math.h>
#include <immintrin.h>
#include <cassert>

#define N 4096
#define K 3
#define ll long long
#define NUM_WORKERS 8
#define BLOCK 8

using namespace std;

// slows the performance down !
//vector<vector<float>> A(N, vector<float>(N,1));
//vector<vector<float>> B(N, vector<float>(N,1));
//vector<vector<float>> res(N, vector<float>(N,0));

float A[N][N];
float B[K][K];
float res[N-K+1][N-K+1];
float val[N-K+1][N-K+1];

float AA[N*N];
float BB[K*K];
float RES[(N-K+1) * (N-K+1)];
float VAL[(N-K+1) * (N-K+1)];

__m256 *Am = (__m256*)AA;
__m256 *Bm = (__m256*)BB;
__m256 *resm = (__m256*)RES;

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

void conv(){
  for(int i=0; i<N-K+1; i++){
    for(int j=0; j<N-K+1; j++){
      float temp = 0;
      for(int x=0; x<K; x++){
        for(int y=0; y<K; y++){
          temp += A[i+x][j+y] * B[x][y];
        }
      }
      res[i][j] = temp;
    }
  }
  for(int i=0; i<N-K+1; i++){
    for(int j=0; j<N-K+1; j++){
      if (fabsf(res[i][j]-val[i][j]) > 1e-3){
        printf("MISMATCH AT %d, %d :: %f != %f",i,j,res[i][j], val[i][j]);
      }
    }
  }
}

void unrolled_conv(){
  for(int i=0; i<(N-K+1); i++){
    for(int j=0; j<(N-K+1); j++){
      float temp = 0;
      for(int x=0; x<K; x++){
        for(int y=0; y<K; y++){
          temp += AA[(x*N)+(i*N) + j+y] * BB[(x*K) + y];
        }
      }
      RES[i*(N-K+1) + j] = temp;
    }
  }
  for(int i=0; i<(N-K+1); i++){
    for(int j=0; j<(N-K+1); j++){
      if (fabsf(RES[i*(N-K+1) + j]-VAL[i*(N-K+1) + j]) > 1e-3){
        printf("MISMATCH AT %d, %d, %f :: %f",i,j,RES[i*(N-K+1) + j], VAL[i*(N-K+1) + j]);
      }
    }
  }
}

// improve space complexity and store operations
void par_conv(){
  int i = 0,remain = 0, RR=0;
  while(i < (N-K+1)){
    int j = remain;
    while(j < (N-K+1)){
      float temp[NUM_WORKERS] = {0};
      #pragma omp parallel for
      for(int id=0; id<NUM_WORKERS; id++){
        for(int x=0; x<K; x++){
          for(int y=0; y<K; y++){
            temp[id] += AA[(x*N)+((i + int(id/(N-K+1))) * N) + j + (id%(N-K+1)) + y] * BB[(x*K) + y];
          }
        }
      }
      for(int x=0; x< (NUM_WORKERS % ((N-K+1)*(N-K+1) - (i*NUM_WORKERS + j*NUM_WORKERS)) == 0 ? (N-K+1)*(N-K+1) : NUM_WORKERS); x++){
        RES[i*(N-K+1)+j+x] = temp[x];
        //printf("%f\n",temp[x]);
      }
      j += NUM_WORKERS;
      remain = (j % (N-K+1));
      if(remain == 0){
        RR++;
      }
    }
    i += int(NUM_WORKERS / (N-K+1)) + RR;
  }
 // for(int i=0; i<(N-K+1); i++){
 //   for(int j=0; j<(N-K+1); j++){
 //     if (fabsf(RES[i*(N-K+1) + j]-VAL[i*(N-K+1) + j]) > 1e-3){
 //       printf("MISMATCH AT %d, %d, %f :: %f \n",i,j,RES[i*(N-K+1) + j], VAL[i*(N-K+1)+j]);
 //     }
 //   }
 // }
}

void fma_conv(){
  for(int i=0; i<(N-K+1); i++){
    for(int j=0; j<(N-K+1); j++){
      __m256 temp = {};
      for(int x=0; x<K; x++){
        for(int y=0; y<K; y++){
          const float* inp = AA + (i+x) * N + (y+j);
          const float* kern = BB + x * K + j;
          __m256 input_data = _mm256_broadcast_ss(inp);
          __m256 kernel_data = _mm256_broadcast_ss(kern);
          temp = _mm256_fmadd_ps(input_data, kernel_data, temp);
        }
      }
      //RES[i*(N-K+1) + j] = temp;
      _mm256_storeu_ps(RES + i * (N-K+1) + j, temp);
    }
  }
  for(int i=0; i<(N-K+1); i++){
    for(int j=0; j<(N-K+1); j++){
      if (fabsf(RES[i + j]-VAL[i + j]) > 1e-3){
        printf("MISMATCH AT %d, %d, %f :: %f",i,j,RES[i + j], VAL[ + j]);
      }
    }
  }
}

void winoconv(){
  return;
}

int main(){

  FILE *f = fopen("./tmp/data","rb");
  fread(AA, 1, sizeof(float)*N*N, f);
  fread(BB, 1, sizeof(float)*K*K, f);
  fread(VAL, 1, sizeof(float)*(N-K+1)*(N-K+1), f);
  fclose(f);

  uint64_t start = nanos();
  par_conv();
  uint64_t end = nanos();
  double time = double(end-start)*1e-9;
  double flop = (N-K+1)*(N-K+1)*(K*K*2.0*K)*1e-9;
  printf("GFLOP/s: %f\n",flop/time);

}

