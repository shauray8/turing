#include <iostream>
#include <vector>
#include <cmath>

// Function to generate the F matrix for Winograd convolution
std::vector<std::vector<double>> generateFMatrix(int N, int K, int M) {
    std::vector<std::vector<double>> F(M + K - 1, std::vector<double>(N, 0.0));
    double alpha = 1.0 / std::sqrt(K * M);

    for (int i = 0; i < M + K - 1; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i < K && i + j < N) {
                F[i][j] = alpha;
            } else if (i >= K && i - K + 1 + j < N) {
                F[i][j] = alpha * (1.0 - (i - K + 1));
            }
        }
    }

    return F;
}

// Function to generate the G matrix for Winograd convolution
std::vector<std::vector<double>> generateGMatrix(int N, int K, int M) {
    std::vector<std::vector<double>> G(K + M - 1, std::vector<double>(K, 0.0));
    double beta = 1.0 / std::sqrt(K * M);

    for (int i = 0; i < K + M - 1; ++i) {
        for (int j = 0; j < K; ++j) {
            if (i < K && i + j < K) {
                G[i][j] = beta;
            } else if (i >= K && i - K + 1 + j < K) {
                G[i][j] = beta * (1.0 - (i - K + 1));
            }
        }
    }

    return G;
}

// Function to perform Winograd convolution
std::vector<std::vector<double>> winogradConvolution(const std::vector<std::vector<double>>& input,
                                                    const std::vector<std::vector<double>>& kernel,
                                                    const std::vector<std::vector<double>>& F,
                                                    const std::vector<std::vector<double>>& G) {
    int N = input.size();  // Input size
    int K = kernel.size(); // Kernel size
    int M = input.size() - kernel.size() + 1; // Output size

    // Perform F-transform on input
    std::vector<std::vector<double>> transformedInput(M + K - 1, std::vector<double>(N, 0.0));
    for (int i = 0; i < M + K - 1; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                transformedInput[i][j] += F[i][k] * input[k][j];
            }
        }
    }

    // Perform G-transform on kernel
    std::vector<std::vector<double>> transformedKernel(K + M - 1, std::vector<double>(K, 0.0));
    for (int i = 0; i < K + M - 1; ++i) {
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < K; ++k) {
                transformedKernel[i][j] += G[i][k] * kernel[k][j];
            }
        }
    }

    // Perform element-wise multiplication in Winograd domain
    std::vector<std::vector<double>> multiplied(M + K - 1, std::vector<double>(M + K - 1, 0.0));
    for (int i = 0; i < M + K - 1; ++i) {
        for (int j = 0; j < M + K - 1; ++j) {
            multiplied[i][j] = transformedInput[i][j] * transformedKernel[i][j];
        }
    }

    // Perform inverse F-transform on multiplied values
    std::vector<std::vector<double>> output(M, std::vector<double>(M, 0.0));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int k = 0; k < K; ++k) {
                output[i][j] += 1.0/F[i][k] * multiplied[k][j];
            }
        }
    }

    return output;
}

int main() {
    // Example usage
    std::vector<std::vector<double>> input = {
        {3, 2, 3},
        {4, 5, 6},
        {7, 8, 1}
    };
    std::vector<std::vector<double>> kernel = {
        {1, 0},
        {0, 0}
    };

    int N = input.size();  // Input size
    int K = kernel.size(); // Kernel size
    int M = input.size() - kernel.size() + 1; // Output size

    std::vector<std::vector<double>> F = generateFMatrix(N, K, M);
    std::vector<std::vector<double>> G = generateGMatrix(N, K, M);

    std::vector<std::vector<double>> output = winogradConvolution(input, kernel, F, G);

    // Print the output
    std::cout << "Output:\n";
    for (const auto& row : output) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

