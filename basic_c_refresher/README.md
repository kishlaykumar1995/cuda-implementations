# Important points regarding C++

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
