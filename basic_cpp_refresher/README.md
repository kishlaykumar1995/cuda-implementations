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

- A **magic number** is a literal that appears in code without explanation or context. Magic numbers can make code difficult to read and maintain, as their meaning is not clear. It is generally considered good practice to avoid magic numbers and use named constants instead.

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

- Using the `const` keyword to create a variable that can be used as a constexpr has some challenges:
  - use of `const` doesn't make it immediately clear if the variable is usable a constant expression or not. This is because `const` variables can be initialized with non-constant expressions, and the compiler will not evaluate them at compile time.
  - use of `const` doesn't inform the compiler that the variable is usable as a constant expression. This means that it will not halt compilation if the variable is not usable as a constant expression and will silently create the variable as a runtime variable.
  - `const` cannot be used to create non-integral constant expression variables. 

- `constexpr` variables are compile-time constants and must be initialized with a constant expression. 

- Even functions that return a constant value cannot be used as constant expressions because they are not evaluated at compile time.

- `constexpr` is not part of the type system. It is a compile-time evaluation mechanism. This means that `constexpr` variables can be used in contexts where a constant expression is required, but they are not part of the type system and do not have any special meaning outside of that context. `constexpr` variables are implicitly `const` type, but `const` variables are not necessarily `constexpr`.

 - `constexpr` functions are functions that can be called constant expressions. It must evaluate at compile time if the constant expression it is a part of must evaluate at compile time. Otherwise, it may be evaluated at runtime.

 - C style strings are arrays of characters terminated by a null character (`'\0'`). They are not automatically resized and must be manually managed. 
 
 - C++ strings are objects of the `std::string` class.They can be automatically resized as needed since they use dynamic memory allocation.

 - `std::string` doesn't directly work with  cin `operator>>` because the operator breaks on whitespace. To read a whole line, use `std::getline(std::cin>>std::ws, string_variable_name)`. `std::ws` is an input manipulator used to ignore any leading whitespace.
 
 - C++ strings also provide many useful member functions for manipulating strings, such as `length()`, `substr()`, and `find()`. (Member functions are functions that are defined inside a class and can be called on objects of that class.)

 - `std::string` supports move semantic operations, which means that it can transfer ownership of its resources to another `std::string` object without copying the data. This can allow us to return string values without the expensive copying of data.

 - `std::string` style literals can be created using 's' suffix. For example, `std::string s = "Hello"s;` creates a string literal of type `std::string`. The suffix 's' is lives in the namespace std::literals::string_literals. We can use `using namespace std::string_literals;` to avoid loading the entire namespace std::literals.

 - `std::string_view` provides read-only access to an existing string without copying it. 

 - `std::string_view` can be initialized with a string literal, `std::string`, or `std::string_view`.

 - `std::string_view` does not implicitly convert to `std::string`. This is because `std::string_view` is a lightweight, non-owning view of a string, while `std::string` is a full-fledged string object that manages its own memory.

 - `std::string_view` literals can be created using the suffix 'sv'. For example, `std::string_view sv = "Hello"sv;` creates a string literal of type `std::string_view`. The suffix 'sv' is lives in the namespace std::string_view_literals.

 - Unlike `std::string`, `std::string_view` can be used with constexpr variables which makes it the preferred choice for string symbolic constants.

 - `std::string` is an owner whereas `std::string_view` is viewer. This means that `std::string_view` does not own the string data it points to, so it is important to ensure that the underlying string data remains valid for the lifetime of the `std::string_view`. If the underlying string data is destroyed or goes out of scope, the `std::string_view` will become invalid and accessing it will result in undefined behavior.

 - Clang compiler evaluates expressions from left to right, while GCC compiler evaluates expressions from right to left. This can lead to different results for the same expression in different compilers. For example, `printCalculation(getValue(), getValue(), getValue());` may produce different results in different compilers. To avoid this, it is best to use temporary variables to store the results of function calls before passing them to another function.

 - A **modifying operator** is an operator that modifies the value of its operand. For example, `++` and `--` are modifying operators because they change the value of the variable they are applied to. Overloaded operators can be redefined to have different behavior than their default behavior which may include modifying the operand.

 - Don't use a variable that has a side effect applied to it more than once in a statement. For example, `i + ++i` is undefined behavior because the order of evaluation is not guaranteed. When initialized with 1, in Clang it would output **1+2 = 3** and in GCC it would output **2+2 = 4**.

 - The `,` operator first evaluates the left operand and then evaluates the right operand. The result of the `,` operator is the value of the right operand. This means that the left operand is evaluated for its side effects, but its value is discarded. For example, in `int a = (b++, c);`, `b` is incremented, but its value is not used. The value of `c` is assigned to `a`. (Mostly used in for loops)

 The **conditional operator**, also called the **arithmetic if** operator, is a ternary operator that takes three operands. It has the form `condition ? expression1 : expression2`. If the condition is true, `expression1` is evaluated and returned; otherwise, `expression2` is evaluated and returned.

 - Floating point numbers have rounding errors when computed and can lead to unexpected results especially when using equality or inequality operators. Generally, equality is tested using a small epsilon value. For example, `if (abs(a - b) < epsilon)`.

 - **Short circuit evaluation** is a technique used for optimizing logical expressions. In short circuit evaluation, the second operand is not evaluated if the first operand is sufficient to determine the result of the expression. For example, in the expression `a && b`, if `a` is false, `b` is not evaluated because the result of the expression is already known to be false. This can improve performance and prevent unnecessary evaluations. This is also why we must not use expressions with side effects in compound expressions. For example, `if (a && ++b)` will not increment `b` if `a` is false.

- Logical AND`&&` and Logical OR`||` do not have the same precedence so it is advisable to put parentheses around them to avoid confusion. For example, `if (a || b && c)` is not the same as `if ((a || b) && c)`. The first expression will evaluate `b&&c` first and then evaluate `a || (result of b&&c)`. The second expression will evaluate `a || b` first and then evaluate `(result of a || b) && c`.

- De Morgan's laws are a pair of transformation rules that are used to simplify logical expressions. They state that:
  - `!(A && B)` is equivalent to `!A || !B`
  - `!(A || B)` is equivalent to `!A && !B`

- C++ has no logical XOR operator. However, we can use the `!=` operator to achieve the same effect. For example, `a != b` is equivalent to `!(a && b) && (a || b)`.

- `std::bitset` allows us to do bit level operations on a fixed number of bits. It contains functions like `set()`, `reset()`, `flip()`, `count()`, `any()`, `none()`, `all()`, and `test()`.

- It is best to use bitwise operators with unsigned integers or bitsets. 

- Bit shifting in C++ is endian agnostic. Left-shift is always towards the most significant bit and right-shift is always towards the least significant bit.

- The `<<` and `>>` operators are overloaded for `std::ostream` and `std::istream` to allow for easy input and output of data.

- Bitwise operators perform integral promotion on their operands. This means that if the operands are smaller than `int`, they will be promoted to `int` before the operation is performed. eg `unsigned short` gets promoted to `unsigned int` before the operation is performed.

- Avoid performing bit shift operations on operands smaller than int because some operators like `~` and `<<` are width sensitive and may produce unexpected results.

- 