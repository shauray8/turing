#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>

__global__ void add_int(int* a, int* b, int count){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < count){
    a[id] *= b[id];
  }
}

void normal_stuff(int* a, int* b, int count){
  for(int i=0; i<count; i++){
    a[i] *= b[i];
  }
}

int main(){
  srand(time(NULL));
  int count = 4096;
  int *h_a = new int[count];
  int *h_b = new int[count];

  for(int i=0; i<count; i++){
    h_a[i] = rand()%1000;
    h_b[i] = rand()%1000;
  }
  
  int *d_a, *d_b;

  if(cudaMalloc(&d_a, sizeof(int)*count) != cudaSuccess){
    printf("take me to the moon BITCH ! A");
    return 0;
  }
  if(cudaMalloc(&d_b, sizeof(int)*count) != cudaSuccess){
    printf("take me to the moon BITCH ! B");
    cudaFree(d_a);
    return 0;
  }

  if(cudaMemcpy(d_a, h_a, sizeof(int)*count, cudaMemcpyHostToDevice) != cudaSuccess){
    printf("fly me to the sky !! ");
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
  }
  if(cudaMemcpy(d_b, h_b, sizeof(int)*count, cudaMemcpyHostToDevice) != cudaSuccess){
    printf("fly me to another sky !! ");
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
  }

  add_int<<<count / 256 + 1, 256>>>(d_a, d_b, count);
  //normal_stuff(h_a, h_b, count);
  
  if(cudaMemcpy(h_a, d_a, sizeof(int)*count, cudaMemcpyDeviceToHost) != cudaSuccess){
    printf("make me a pancake !! ");
    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
  }

  for(int j=0;j<5;j++){
    printf("the answer is : %d %d\n",h_a[j], h_b[j]);
  }

  cudaFree(d_a);
  cudaFree(d_b);

  delete[] h_a;
  delete[] h_b;

  cudaDeviceReset();

  return 0;
  
}
