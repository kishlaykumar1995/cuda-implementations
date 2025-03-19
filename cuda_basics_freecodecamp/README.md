# CUDA Basics

This folder contains some basic theory on CUDA Architecture and some basic parallelisation examples which demonstrate the basic concepts of CUDA programming, including kernel launches, memory management, and synchronization.
Refer to the freeCodeCamp video [here](https://www.youtube.com/watch?v=86FAWCzIe_4&t=3s) for more information.

## Theory
- **CUDA (Compute Unified Device Architecture)**: A parallel computing platform and programming model developed by NVIDIA.
    - Enables developers to use NVIDIA GPUs for general-purpose processing.

- **Some Teminology**:
    - **Host**: The CPU and its memory.
    - **Device**: The GPU and its memory.
    - **Kernel**: A function that runs on the GPU and is executed by multiple threads in parallel.
    - **Thread**: The smallest unit of execution in CUDA, executing a kernel.
    - **Block**: A group of threads that can cooperate with each other through shared memory.
    - **Grid**: A collection of blocks that execute a kernel.
    - **Warp**: A group of 32 threads that execute in lockstep on the GPU.

- A CUDA program typically consists of:
    - **Host code**: Runs on the CPU and manages memory, kernel launches, and data transfers. Copies input data to the device, launches the kernel, and copies the output data back to the host.
    - **Device code**: Runs on the GPU and performs parallel computations.

- **Naming Schemes**:
    - `__global__`: A function that runs on the device and can be called from the host. It is executed by multiple threads in parallel.
    - `__device__`: A function that runs on the device and can be called from other device functions or kernels.
    - `__host__`: A function that runs on the host and can be called from other host functions or kernels. Same as running functions in C/C++.

- **Memory Management**:
    - `cudaMalloc()`: Allocates memory on the device. (Global memory on the GPU) eg. `cudaMalloc((void**)&d_a, size);`
    - `cudaMemcpy()`: Copies data between host and device memory. (Host to Device or Device to Host) eg. `cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);`
        - `cudaMemcpyHostToDevice`: Copies data from host to device.
        - `cudaMemcpyDeviceToHost`: Copies data from device to host.
        - `cudaMemcpyDeviceToDevice`: Copies data from device to device.
    - `cudaFree()`: Frees memory on the device. eg. `cudaFree(d_a);`

- **nvcc compiler**:
    - The NVIDIA CUDA Compiler (nvcc) is used to compile CUDA code. It separates the host and device code and compiles them accordingly.
    - Host code:
        - Compiled and runs on the CPU. It calls the device code.
        - Compiled to standard x86 binary.
    - Device code:
        - Compiled to PTX (Parallel Thread Execution) code which is an intermediate representation (like JAVA bytecode).
        - PTX code is then compiled to binary code for the target GPU architecture by the target JIT (Just-In-Time) compiler.

- **CUDA Hierarchy**:
    - Threads are organized into blocks.
    - Blocks are organized into a grid.
    - Each block can contain up to 1024 threads (depending on the GPU architecture).
    - Each thread has a unique thread ID within its block.
    - Each block has a unique block ID within the grid.
    - The grid can be 1D, 2D, or 3D.
    - The block can be 1D, 2D, or 3D. 
    - The thread ID and block ID can be used to calculate the global thread ID, which is unique across the entire grid.
    - Kernel executes in parallel across threads in a block and across blocks.
    - Threads can communicate with each other within a block using shared memory.

- **CUDA Inbuilt Variables**:
    - `blockIdx`: The block index within the grid.
    - `blockDim`: The dimensions of the block (number of threads in each dimension).
    - `threadIdx`: The thread index within the block.
    - `gridDim`: The dimensions of the grid (number of blocks in each dimension).
    - All of these variables can have x, y, and z components (1D, 2D, or 3D).

- **Threads**:
    - Each one has its own registers and local memory.
    - If we wanted to add two vectors of size N, we could launch N threads, each responsible for adding one element of the vectors.
    - Each thread would have its own index, which it could use to access the corresponding elements of the vectors.

- **Warps**:
    - A warp is a group of 32 threads that execute in lockstep on the GPU.
    - Warp schedulers manage the execution of warps on the GPU.
    - 4 warps schedulers per SM (Streaming Multiprocessor) on NVIDIA GPUs.

- **Blocks**:
    - A block is a group of threads that can cooperate with each other through shared memory called the L1 Cache.
    - Blocks execute the same kernel code but can have different thread IDs which is used to access different data.

- **Grids**:
    - Contains blocks of threads.
    - During kernel execution threads within the blocks within the grids can access global memory (VRAM) but it is slower than shared memory.
    

## Basic Addition

| Blocks  | Threads | Runtime (ms) |
|---------|---------|--------------|
| 1       | 1       | 246          |
| 1       | 128     | 11           |
| 8192    | 128     | 3.2          |
| 4096    | 256     | 2.9          |
