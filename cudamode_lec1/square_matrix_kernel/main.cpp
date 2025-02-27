#include <torch/extension.h>
torch::Tensor square(torch::Tensor matrix);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("square", torch::wrap_pybind_function(square), "square");
}