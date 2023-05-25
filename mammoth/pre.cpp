#include <iostream>
#include <immintrin.h>


int main(){

  float X[] = {1.,2.,3.,4.,5.,6.,7.,18.};
  float Y[] = {1.,2.,3.,4.,5.,6.,7.,8.};
  float Z[] = {1.,2.,3.,4.,5.,6.,7.,8.};
  float result;
  
  
  __m256 a = _mm256_broadcast_ss(X);
  __m256 b = _mm256_broadcast_ss(Y);
  __m256 c = _mm256_broadcast_ss(Z);
  __m256 res = _mm256_fmadd_ps(a,b,c);

  _mm256_storeu_ps(&result, res);
  printf("%f",result);
}
