#include <bitset>
#include <iostream>

std::bitset<4> rotl(std::bitset<4> bits) {
    /*
    Right Shift bits by 3 (to get msb to right)
    Left shift bits by 1
    Perform Logical OR b/w left shifted bits and right shifted bits
    */
    std::bitset<4> msb{bits >> 3};
    bits<<=1;
    return bits | msb;
}

int main() {
    /*
    Perform left rotate without using bitset member functions
    */
    std::bitset<4> bits{0b1011};
    std::cout << rotl(bits) << '\n';
}
