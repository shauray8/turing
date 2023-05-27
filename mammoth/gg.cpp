#include <iostream>
#include <immintrin.h> // Header for SIMD intrinsics

int64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

int inputWidth = 1024;
int inputHeight = 1024;
int kernelSize = 3;
int outputWidth = inputWidth - kernelSize + 1;
int outputHeight = inputHeight - kernelSize + 1;
#define N 1024
#define K 3

float A[N*N];
float B[K*K];
float RES[(N-K+1) * (N-K+1)];
float VAL[(N-K+1) * (N-K+1)];

// 2D Convolution using SIMD instructions
void convolution2DSIMD(float* input, float* kernel, float* output, int inputWidth, int inputHeight, int kernelSize)
{
    int outputWidth = inputWidth - kernelSize + 1;
    int outputHeight = inputHeight - kernelSize + 1;
    
    // Process the input and kernel arrays in chunks of 8 elements
    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            __m256 sum = {}; // Initialize sum to zero
            
            for (int k = 0; k < kernelSize; ++k) {
                for (int l = 0; l < kernelSize; ++l) {
                    // Calculate the indices in the input and kernel arrays
                    int inputIndex = (i + k) * inputWidth + (j + l);
                    int kernelIndex = k * kernelSize + l;
                    
                    // Load 8 elements from the input array
                    __m256 inputVec = _mm256_broadcast_ss(input + inputIndex);
                    
                    // Load 8 elements from the kernel array
                    __m256 kernelVec = _mm256_broadcast_ss(kernel + kernelIndex);
                    
                    // Multiply corresponding elements of input and kernel vectors
                    sum = _mm256_fmadd_ps(inputVec, kernelVec, sum);
                    
                    // Accumulate the results in the sum vector
                }
            }
            
            // Store the sum vector into the output array
            _mm256_storeu_ps(RES + i * outputWidth + j, sum);
        }
    }
}


int main()
{

    FILE *f = fopen("./tmp/data","rb");
    fread(A, 1, sizeof(float)*N*N, f);
    fread(B, 1, sizeof(float)*K*K, f);
    fread(VAL, 1, sizeof(float)*(N-K+1)*(N-K+1), f);
    fclose(f);

    float input[] = {1.0, 2.0, 3.0, 4.0,
                     5.0, 6.0, 7.0, 8.0,
                     9.0, 10.0, 11.0, 12.0,
                     13.0, 14.0, 15.0, 16.0};
                     
    float kernel[] = {0.5, 0.5, 0.5,
                      0.5, 0.5, 0.5,
                      0.5, 0.5, 0.5};
                      
    float output[outputWidth * outputHeight];

    for(int i=0; i<(N-K+1); i++){
      for(int j=0; j<(N-K+1); j++){
        float temp = 0;
        for(int x=0; x<K; x++){
          for(int y=0; y<K; y++){
            temp += A[(x*N)+(i*N) + j+y] * B[(x*K) + y];
          }
        }
        RES[i*(N-K+1) + j] = temp;
      }
    }

    //for (int i = 0; i < outputHeight; ++i) {
    //    for (int j = 0; j < outputWidth; ++j) {
    //        std::cout << RES[i * outputWidth + j] << " -----";
    //    }
    //    std::cout << std::endl;
    //}

    int64_t start = nanos();
    convolution2DSIMD(A, B, RES, inputWidth, inputHeight, kernelSize);
    int64_t end = nanos();
    double time = double(end-start)*1e-9;
    double flop = (N-K+1)*(N-K+1)*(K*K*2.0*K)*1e-9;
    printf("GFLOP/s: %f\n",flop/time);
    
    // Print the output
   // for (int i = 0; i < outputHeight; ++i) {
   //     for (int j = 0; j < outputWidth; ++j) {
   //         std::cout << RES[i * outputWidth + j] << " ";
   //     }
   //     std::cout << std::endl;
   // }
    
    return 0;
}

