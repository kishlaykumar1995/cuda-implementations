#include <iostream>
#include <iomanip>

int main() {
    float f = 13.48f;
    // Use set precision to see the slight inaccuracies in how floating points are actually stored
    std::cout << std::setprecision(32)<< f << "\n";
    return 0;
}