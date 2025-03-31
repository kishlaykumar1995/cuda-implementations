#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

__global__ void matmul_gpu(float *matA, float *matB, float *matC, int M, int K, int N) {
    /*
    Here we compute the matrix multiplication:
    - The x dimension represents the row in our output matrix (and matA)
    - The y dimension represents the column in our output matrix (and matB)
    - For each position in our output matrix, the dot product of the respective vectors in matA and
      matB are calculated independently on a seperate thread.
    */
    int x_Idx = blockIdx.x*blockDim.x + threadIdx.x;
    int y_Idx = blockIdx.y*blockDim.y + threadIdx.y;
    if(x_Idx < M && y_Idx < N) {                        // This is to avoid unnecessary computation on 
                                                         // out of bounds threads
        float sum = 0.0f;
        for(int k=0; k<K; k++) {
            sum+=matA[x_Idx*K + k]*matB[k*N + y_Idx];
        }
        matC[x_Idx*N + y_Idx] = sum;
    }
}

void matmul_cpu(float *matA, float *matB, float *matC, int M, int K, int N) {
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            for(int k=0;k<K;k++) {
                matC[i*N + j]+=matA[i*K + k]*matB[k*N + j];
            }
        }
    }
}

void init_matrix(float *matrix, int rows, int cols, float seed) {
    for(int i=0;i<rows;i++) {
        for(int j=0;j<cols;j++) {
            matrix[i*cols+j] = seed*((float)rand())/RAND_MAX;
        }
    }
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;

    // Define matrix dimensions
    int M = 256;                            // No of rows in A and C
    int K = 512;                            // No of columns in A and rows in B
    int N = 256;                            // No of columns in B and C
    int blockSize = 32;
    size_t sizeA = M*K*sizeof(float);
    size_t sizeB = K*N*sizeof(float);
    size_t sizeC = M*N*sizeof(float);

    // Allocate memory
    h_a = (float *) malloc(sizeA);
    h_b = (float *) malloc(sizeB);
    h_c_cpu = (float *) malloc(sizeC);
    h_c_gpu = (float *) malloc(sizeC);
    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_b, sizeB);
    cudaMalloc(&d_c, sizeC);

    // initialize matrix
    init_matrix(h_a, M, K, 1.0f);
    init_matrix(h_b, K, N, 2.0f);
    init_matrix(h_c_cpu, M, N, 0.0f);
    init_matrix(h_c_gpu, M, N, 0.0f);

    // Copy matrices to device
    cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c_gpu, sizeC, cudaMemcpyHostToDevice);

    // Set kernel parameters
    dim3 numThreads(blockSize, blockSize);
    dim3 numBlocks((M+blockSize-1)/blockSize, (N+blockSize-1)/blockSize);    

    // Launch kernel
    matmul_gpu<<<numBlocks, numThreads>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c_gpu, d_c, sizeC, cudaMemcpyHostToDevice);

    // Do the matmul on cpu
    matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);

    // Compare the results
    bool MATCH_FLAG = true;
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            if(fabs(h_c_gpu[i*M + j] - h_c_cpu[i*M + j]) > 1e-5) {
                std::cout << "CPU: " << h_c_cpu[i*M + j] << ", GPU: " << h_c_gpu[i*M + j] << std::endl;
                MATCH_FLAG = false;
                break;
            }
        }
        if(!MATCH_FLAG)
            break;
    }
    if (MATCH_FLAG) {
        std::cout << "Results match";
    }
    else {
        std::cout << "Results don't match";
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}