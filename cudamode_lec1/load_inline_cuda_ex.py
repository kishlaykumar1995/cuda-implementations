import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
     int row = blockIdx.y*blockDim.y + threadIdx.y;
     int column = blockIdx.x*blockDim.x + threadIdx.x;

     if(row<height && column<width) {
        int idx = row*width + column;
        result[idx] = matrix[idx] * matrix[idx];
     }
}

torch::Tensor square(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x-1)/threads_per_block.x,
                          (height + threads_per_block.y-1)/threads_per_block.y);
    
    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
                        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);
    
    return result;
}
"""
cpp_source = "torch::Tensor square(torch::Tensor matrix);"

square_matrix_extension = load_inline(
    name="square_matrix_ext",
    cpp_sources=[cpp_source],
    cuda_sources=cuda_source,
    verbose=True,
    functions=["square"],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory="./square_matrix_kernel"
)

a = torch.tensor([[1., 2., 3.],[4., 5., 6.]], device='cuda')
print(square_matrix_extension.square(a))