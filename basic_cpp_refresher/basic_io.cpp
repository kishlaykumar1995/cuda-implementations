#include <stdio.h>
#include <iostream>  //iostream is the standard input/output library in C++ (iostream.h is the old version)

int main() {
    std::cout << "Enter a number: ";
    int number;
    std::cin >> number;
    std::cout << "You entered: " << number << std::endl;
    // printf("You entered: %d\n", number);
    return 0;
}
