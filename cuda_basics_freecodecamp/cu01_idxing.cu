#include <iostream>

__global__ void calculateMyPosition(void) {
    /*
    This is basically is flattened out representation of the 3-D Cubes that blocks and grids represent.
    To find the global id of our location we need to see:
    1) How far we have moved in the grid (block_id): 
        - For each step we move in the x direction we move 1 block
        - For each step we move in the y direction we move gridDim.x blocks 
            (After completing the x direction)
        - For each step we move in the z direction we move gridDim.x * gridDim.y blocks 
            (After completing the y direction (which in turn completed the x direction))
        - So, the formula for block_id is:
            block_id = blockIdx.x + blockIdx.y * gridDim.x  + blockIdx.z * gridDim.y * gridDim.x

    2) How many threads have we moved past to reach our current block (block_offset):
        - For each new block_id we move past we move blockDim.x * blockDim.y * blockDim.z threads
        - So, the formula for block_offset is:
            block_offset = block_id * blockDim.x * blockDim.y * blockDim.z

    3) How far we have moved in the block (thread_offset):
        - For each step we move in the x direction we move 1 thread
        - For each step we move in the y direction we move blockDim.x threads 
            (After completing the x direction)
        - For each step we move in the z direction we move blockDim.x * blockDim.y threads 
            (After completing the y direction (which in turn completed the x direction))
        - So, the formula for thread_offset is:
            thread_offset = threadIdx.x + threadIdx.y * blockDim.x  + threadIdx.z * blockDim.y * blockDim.x
    
    4) Finally, we can find the global id by adding the block_offset and thread_offset:
        id = block_offset + thread_offset
    
    See the apartment building analogy on the freecodecamp CUDA course for a more intuitive explanation.
    */
    int block_id = 
        blockIdx.x + 
        blockIdx.y * gridDim.x  + 
        blockIdx.z * gridDim.y * gridDim.x;
    
    int block_offset = 
        block_id * blockDim.x * blockDim.y * blockDim.z;
    
    int thread_offset = 
        threadIdx.x + 
        threadIdx.y * blockDim.x  + 
        threadIdx.z * blockDim.y * blockDim.x;
    
    int id = block_offset + thread_offset;

    // The output of this program will have indexes from 0-31 and then jump 
    // which indicates how threads are executed in warps of 32.
    printf("%07d | Block (%d %d %d) = %3d | Thread (%d %d %d) = %3d\n",
            id,
            blockIdx.x, blockIdx.y, blockIdx.z, block_id, 
            threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);

}

int main(int argc, char **argv) {
    const int b_x = 2, b_y = 3, b_z = 4;       // grid dimensions (No of blocks per grid)
    // Since warps are in groups of 32 and execute in lockstep, so it is good to have block dimensions as 
    // multiples of 32 so that all threads are utilized. Here, we have 2 warps of 32 threads per block
    const int t_x = 4, t_y = 4, t_z = 4;       // block dimensions (No of threads per block)

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d threads/block\n", threads_per_block);
    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d total threads\n", threads_per_block*blocks_per_grid);

    dim3 blocksPerGrid(b_x, b_y, b_z);      // 3-D cube 2*3*4
    dim3 threadsPerBlock(t_x, t_y, t_z);    // 3-D cube 4*4*4

    calculateMyPosition<<<blocksPerGrid, threadsPerBlock>>>(); 
    cudaDeviceSynchronize();

    return 0;
}