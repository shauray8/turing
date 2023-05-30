#include <iostream>
#include <vector>

std::vector<std::vector<double>> winogradConvolution(const std::vector<std::vector<double>>& input,
                                                    const std::vector<std::vector<double>>& kernel) {
    int N = input.size();        // Input size
    int K = kernel.size();       // Kernel size
    int M = N - K + 1;           // Output size

    // Calculate D matrix
    std::vector<std::vector<double>> D(N+K-1, std::vector<double>(K, 0.0));
    D[0][0] = 1.0;
    D[1][0] = 1.0;
    D[2][0] = 1.0;
    for (int i = 3; i < K + 2; ++i) {
        D[i][0] = -1.0 / (1 << (i - 2));
    }

    // Calculate B matrix
    std::vector<std::vector<double>> B(N+K-1, std::vector<double>(N, 0.0));
    for (int i = 2; i < N + 2; ++i) {
        for (int j = 2; j < N + 2; ++j) {
            B[i][j] = input[i - 2][j - 2];
        }
    }

    // Calculate A matrix
    std::vector<std::vector<double>> A(M, std::vector<double>(N+K-1, 0.0));
    for (int i = 0; i < M; ++i) {
        for (int j = 2; j < M + 2; ++j) {
            A[i][j] = 1.0;
        }
    }

    // Perform Winograd convolution
    std::vector<std::vector<double>> transformedInput(K, std::vector<double>(K, 0.0));
    std::vector<std::vector<double>> transformedKernel(K, std::vector<double>(K, 0.0));
    std::vector<std::vector<double>> transformedOutput(M, std::vector<double>(M, 0.0));

    // Transform input to Winograd domain
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < K; ++k) {
                transformedInput[i][j] += B[i][k] * input[k][j];
            }
        }
    }

    // Transform kernel to Winograd domain
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < K; ++k) {
                transformedKernel[i][j] += D[i][k] * kernel[k][j];  // Fixed the indices here
            }
        }
    }

    // Perform element-wise multiplication in the Winograd domain
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int k = 0; k < K; ++k) {
                for (int l = 0; l < K; ++l) {
                    transformedOutput[i][j] += transformedInput[k][l] * transformedKernel[i + k][j + l];  // Fixed the indices here
                }
            }
        }
    }

    // Transform output back to spatial domain using A matrix
    std::vector<std::vector<double>> output(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < M; ++k) {
                for (int l = 0; l < M; ++l) {
                    output[i][j] += A[k][i] * transformedOutput[k][l] * A[l][j];
                }
            }
        }
    }

    return output;
}

int main() {
    // Example usage
    std::vector<std::vector<double>> input = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::vector<double>> kernel = {
        {1, 0},
        {0, 1}
    };

    std::vector<std::vector<double>> output = winogradConvolution(input, kernel);

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

