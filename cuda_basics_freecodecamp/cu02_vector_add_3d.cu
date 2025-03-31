#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

__global__ void add_3d(int N, float *x, float *y, float *z) {
    /*
    This is a slightly different way of indexing the threads in a 3D grid than the one in cu01_idxing.cu.
    Previously, we calculated a flattened block id in the block idx space, converted into the thread idx 
    space and then added the thread offset to get the global id.
    Here, we convert each dimension directly into the thread idx space and then add them up.
    */
    int idx = threadIdx.x + blockIdx.x*blockDim.x
            + (threadIdx.y + blockIdx.y*blockDim.y) * gridDim.x*blockDim.x
            + (threadIdx.z + blockIdx.y*blockDim.z) * gridDim.y*blockDim.y*gridDim.x*blockDim.x;
    
    if(idx < N) {
        z[idx] = x[idx] + y[idx];
    }
}

void initialize_vector(int N, float *vec, float init_value) {
    for(int i=0; i<N; i++) {
        vec[i] = init_value*((float)rand())/RAND_MAX;
    }
}

int main() {
    int N = 1<<20;
    float *h_a, *h_b, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N*sizeof(float);

    // Allocate memory to host vectors
    h_a = (float *) malloc(size);
    h_b = (float *) malloc(size);
    h_c_gpu = (float *) malloc(size);

    // Initialize host vectors
    initialize_vector(N, h_a, 1.0f);
    initialize_vector(N, h_b, 2.0f);
    initialize_vector(N, h_c_gpu, 0.0f);

    // Allocate memory to device vectors
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size); 

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c_gpu, size, cudaMemcpyHostToDevice);

    // Set kernel parameters
    dim3 numThreads(4, 4, 4);
    dim3 numBlocks(32, 32, 16);    // Or we could do N + numThreads.x -1 / numThreads.x and so on for y and z

    // Launch the kernel
    add_3d<<<numBlocks, numThreads>>>(N, d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    
    for(int i=0; i<10;i++) {
        std::cout <<h_a[i]<<" "<<h_b[i]<<" "<< h_c_gpu[i]<<"\n";
    }
    
    // Free the memory
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}