from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
    return "Hello, World";
}
"""

hello_world_extension = load_inline(
    name="hello_world_ext",
    cpp_sources=[cpp_source],
    verbose=True,
    functions=["hello_world"],
    build_directory="./build_hello_world"
)

print(hello_world_extension.hello_world())