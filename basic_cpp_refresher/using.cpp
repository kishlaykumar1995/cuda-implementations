#include <iostream>

namespace A {
    void foo() {
        std::cout << "This is A" << "\n";
    }
}

namespace B {
    void foo() {
        std::cout << "This is B" << "\n";
    }
}

void foo() {
    std::cout << "This is unnamed foo" << "\n";
}

int main() {
    using A::foo;
    foo();                        // This still calls namespace A foo because we explicitly mention it.
                                  // Hence, no name clash with global foo
    ::foo();                      // To call global foo we can use the scope operator with no name.
    return 0;
}