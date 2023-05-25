#include <iostream>
#include <immintrin.h>
#include <time.h>

int64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

int main(){

  float X[] = {21.,12.,3.,4.,5.,6.,7.,18.};
  float Y[] = {11.,2.1,3.2,4.,5.,6.,7.,8.2};
  float Z[] = {1.,2.,3.3,4.,5.,1.,7.2,81.};
  float result[8];
  
  int64_t start = nanos();
  for(int i=0; i<8; i++){
    result[i] = X[i]*Y[i]+Z[i];
  }
  int64_t end = nanos();
  printf("%ld s \n",end-start);
  for(int i=0; i<8; i++){
    printf("normal %f\n",result[i]);
  }
  
  start = nanos();
  __m256 aa = _mm256_loadu_ps(X);
  __m256 bb = _mm256_loadu_ps(Y);
  __m256 cc = _mm256_loadu_ps(Z);
  //__m256 a = _mm256_broadcast_ss(X);
  //__m256 b = _mm256_broadcast_ss(Y);
  //__m256 c = _mm256_broadcast_ss(Z);
  __m256 res = {};
  res = _mm256_fmadd_ps(aa,bb,cc);

  _mm256_storeu_ps(result, res);

  end = nanos();
  printf("%ld s \n",end-start);
  for(int i=0; i<8; i++){
    printf("%f\n",result[i]);
  }

}
