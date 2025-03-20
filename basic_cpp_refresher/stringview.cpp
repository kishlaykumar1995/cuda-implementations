#include <iostream>
# include <string_view>

int main() {
    // C style string literals exist for the entire life of the program and thus the string_view prints
    // the string literal correctly.
    std::string_view str{"Hello, World!"};
    std::cout << str << std::endl;
    return 0;
}