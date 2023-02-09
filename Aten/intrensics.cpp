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

__m256 A[BLOCK] = {1,2,3,4,5,6,7,8};
__m256 B[BLOCK] = {1,2,3,4,5,6,7,8};


int main(){
  for(int i=0; i<8; i++){
    cout << A[i] << B[i];
  }

}
