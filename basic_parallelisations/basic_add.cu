#include<iostream>
#include<math.h>

__global__ void add(int N, float *x, float *y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for(int i = idx; i < N; i+=stride) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1<<20;
    float *a, *b;
    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    add<<<4096, 256>>>(N, a, b);
    cudaDeviceSynchronize();
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(b[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;
    cudaFree(a);
    cudaFree(b);
    return 0;
}