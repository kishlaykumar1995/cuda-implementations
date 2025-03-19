# Important points regarding C++

[Reference Link](https://www.learncpp.com/)

- C++ is a statically typed language, meaning that the type of a variable must be known at compile time.

- C++ has different types of initialization:
  - Default initialization: `int a;` (Can have garbage value)
  - Copy initialization: `int a = 5;`
  - Direct initialization: `int a(5);`
  - Direct-list-initialization: `int a{5};`  (Narrowing conversion is not allowed)
  - Value initialization: `int a{};`

- C++ does not automatically initialize most variables.

- Operators like `=` and `>>` return the left-hand side, which allows for chaining. For example:
  ```cpp
  int a = 5;
  int b = a = 10; // b is now 10
  cout << "Hello" << " World"; // prints Hello World
  ```

- Variables are most often destroyed in the reverse order of their creation but generally the compilers have a lot of flexibility in this regard.

- Destructors are invoked for class type objects when they go out of scope.

- Forward declarations can be used to tell the compiler about the existence of a class or function before its actual definition.

- One Definition Rule (ODR) states that an entity can only be defined once in a program, but it can be declared multiple times.

- Namespaces provide a scope region to declare and define identifiers inside of it for the purpose of disambiguation.

- Preprocessor Directives are commands that are processed before the compilation of the code. They are used to include files, define macros, and conditionally compile code.

- When working with multiple files, forward declarations can related to a particular functionality can all be included in a single header file.

- As best practice, header files should be included in the paired .cpp files as well. This allows the compiler to check for certain errors at compile time rather than during linking (eg: any changes to return types, etc). However, any changes to parameters are still not detected at compile time because C++ supports function overloading and the inclusion of a differently paramteterized declaration could simply mean that it is defined elsewhere in the codebase. Such errors are only detected at link time.

- Header guards are added to ensure that a header file is only included once in a translation unit. However, the same header can be included in multiple files which include them which can cause multiple definitions of the same function and thus cause linker errors. Thus, we must avoid function definitions in header files. 

- std::cout and std::cin are buffered whereas std::cerr is unbuffered.

- Some common debugger functions:
  - Step into: Move into the function call (gdb command: `step`)
  - Step over: Move to the next line of code (gdb command: `next`)
  - Step out: Move out of the current function (gdb command: `finish`)
  - Continue: Run the program until the next breakpoint (gdb command: `continue`)
  - Breakpoint: A point in the code where the debugger will pause execution (gdb command: `break`)  

- C++ standards only specify the minimum size of each type. The actual size may vary depending on the compiler and architecture. For example, `sizeof(int)` is guaranteed to be at least 2 bytes, but it can be larger.

- Direct comparison using floating point numbers can lead to unexpected results due to precision errors.

- `const` and `volatile` are two keywords that can be used to modify the behavior of variables. `const` indicates that a variable's value cannot be changed after it is initialized, while `volatile` indicates that a variable's value may change at any time, such as when it is modified by an external process or hardware.

- Zero before a number means octal (base 8) representation. For example, `012` is equal to `10` in decimal. Similarly, `0x` before a number means hexadecimal (base 16) representation. For example, `0xA` is equal to `10` in decimal.

- The **as if** rule states that the compiler is free to optimize the code as long as the observable behavior of the program remains the same. This means that the compiler can rearrange, combine, or eliminate instructions as long as the final result is the same.

- **Compile time evaluation** is the process of evaluating expressions at compile time rather than at runtime. 

- Debug builds typically turn optimizations off because they can make debugging difficult. Optimizations can change the order of operations, inline functions, and remove unused code, making it harder to trace the flow of execution. Release builds typically turn optimizations on to improve performance.

- Compile time evaluatable expressions is required for `constexpr` variables, non-type template parameters, and for defined length of std::array.

- C++ features foundational to compile time programming:
  - `constexpr` functions and variables
  - Templates
  - static assertions

- Constant expressions can contain:
  - Literal types
  - `constexpr` variables
  - `constexpr` functions
  - Most operators with constant operands
  - Non-type template parameters
  - Enumerators
  - `constexpr` lambda expressions
  - Type traits
  - `const` **integral** variables with constant expression initializers

- Compiler is only required to evaluate constant expressions at compile time in contexts where a constant expression is required. Because for example from the statement `const int x {3+4};`, x is usable as a constant expression and if it is not evaluated at compile time then it will not be usable as a constant expression. 