#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

__global__ void add(int N, float *x, float *y, float *z) {
    /*
    Since, we only work with 1-D arrays, i.e., the x coordinate, so the block_id = blockIdx.x,
    block_offset = gridDim.x * blockIdx.x(or block_id), and thread_offset = threadIdx.x.
    The final index is calculated as:
    idx = threadIdx.x + blockIdx.x * blockDim.x; 

    The stride is calculated as: blockDim.x * gridDim.x
    It is needed only when the number of elements in the array is greater than the number of threads because
    in that case, the threads will be reused to process the remaining elements of the array.
    */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < N; i+=stride) {
        z[i] = x[i] + y[i];
    }
}

void initialize_vector(float *vec, int N, float init_value) {
    for (int i=0; i<N; i++) {
        vec[i] = init_value*((float)rand())/RAND_MAX;
    }
}

int main() {
    int N = 1<<20;   // 2048 elements in vector
    float *h_a, *h_b, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N*sizeof(float);

    // Allocate memory to host vectors
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_gpu = (float *)malloc(size);

    // Initialize host vectors
    initialize_vector(h_a, N, 2.0f);
    initialize_vector(h_b, N, 1.0f);
    initialize_vector(h_c_gpu, N, 0.0f);

    // Allocate memory to device vectors
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy the vectors to the device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c_gpu, size, cudaMemcpyHostToDevice);

    // Set kernel parameters
    // NOTE: Instead of using a dim3 object with x,y and z axes we can also use simple integers. In this 
        // case, it is converted to dim3 (N,1,1) where 1 is N is the number you passed. So, it still 
        // gets converted to 3-D but it is treated as line by CUDA and only x coordinates have values.
    int numThreads = 64;
    // No of blocks required can be calculated as ceil(N/numThreads) or (N+numThreads-1)/numThreads. This
        // ensures that the number of blocks is rounded up to the nearest integer and we have more than 
        // enough blocks to process each element in a seperate thread.
    int numBlocks = (N + numThreads - 1) / numThreads;

    // Launch the kernel
    add<<<numBlocks, numThreads>>>(N, d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Copy back the computed result to host
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    for(int i=0; i<10;i++) {
        std::cout <<h_a[i]<<" "<<h_b[i]<<" "<< h_c_gpu[i]<<"\n";
    }
    
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}