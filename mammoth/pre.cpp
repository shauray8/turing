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

void matmul(){

}

int main(){

}
