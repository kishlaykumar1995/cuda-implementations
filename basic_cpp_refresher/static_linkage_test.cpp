#include <iostream>
/*
Here static means the variable has internal linkage.
Variable g_i will only be accessible in this file (or translation unit)
*/
static int g_i{4};

void add() {
    /*
    Here, static ensures the variable has static durastion.
    The value of i is preserved across function calls in this case.
    */
    static int i=1; // 
    int j=1;
    std::cout << (i+j)<< '\n';
    i++;
}

int main() {
    add();
    add();
    std::cout << g_i << '\n';
    return 0;
}