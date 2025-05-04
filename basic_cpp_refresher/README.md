# Important points regarding C++

Source: [learncpp.com](https://www.learncpp.com/)

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

- Namespaces can be accessed in two ways:
  - Fully qualified name: `std::cout`
  - Using directive: `using namespace std;` (not recommended; can cause name clashes)

- Forward declarations of functions defined in a namespace must be in the same namespace. For example:
  ```cpp
  namespace math {
      int add(int a, int b);
  }
  ```
  ```cpp
  #include "add.h"
  namespace math {
      int add(int a, int b) {
          return a + b;
      }
  }
  ```

- Same namespace blocks can be declared in multiple locations (across files or within the same file) and the compiler will merge them into a single namespace. The C++ standard library makes extensive use of this feature to provide a consistent interface across multiple files. For example, the `std::cout` and `std::cin` objects are declared in multiple header files, but they are all merged into the same `std` namespace when the program is compiled.

- Local variables have **block scope** (i.e., they are only visible within the block in which they are declared) and **automatic duration** (i.e., they are created when the block is entered and destroyed when the block is exited).

- Global variables have **file scope** (or **global scope**) (i.e., they are visible throughout the file in which they are declared) and **static duration** (i.e., they are created when the program starts and destroyed when the program ends).

- Variables defined in namespaces with global scope are also global variables. However, they are accessible only via the namespace they are defined in. For example, `std::cout` is a global variable that is defined in the `std` namespace. It is accessible only via the `std` namespace. 

- Unlike local variables, variables with static duration(including static local variables) are automatically initialized to zero if not explicitly initialized.

- If we have 2 variables of the same name inside a nested block and we try to access it, then the innermost variable will be used and the outer scope variable is hidden. This is called **variable shadowing**. Variable shadowing is best avoided because it can lead to confusion and bugs in the code (like the wrong variable being used/modified).

- If we want to access a global scope variable that is shadowed by a local variable, we can use the `::` operator.

- An identifier's linkage determines whether other declarations of the same identifier refer to the same entity or not. Global (non const) variables have **external linkage** by default, which means that they can be accessed from other translation units. Local variables have no linkage. To make a non-constant global variable have internal linkage, we can use the `static` keyword.

- Functions have external linkage by default, which means that they can be accessed from other translation units. To make a function have internal linkage, we can use the `static` keyword. This is useful for creating helper functions that are only used within a single translation unit.

- `extern` keyword can be used to make global variables and functions have **external linkage**. Non-constant global variables have external linkage by default, so `extern` is not needed. `extern` can also be used to place a **forward declaration** of a variable in a different translation unit (with no initialization). 

- Variables and functions with external linkage can be accessed anywhere within our program via forward declarations. Forward function declarations don't need the `extern` keyword because functions have external linkage by default and the compiler can understand that it is a forward declaration based on whether we give the function body. However, variable forward declarations need the `extern` to distinguish between uninitialized variable definitions and forward declarations (else they look the same).

- Non constant global variables are best avoided for the following reasons:
  - They can be changed by any part of the program, making it difficult to track down bugs.
  - They make the program less modular and harder to understand. Also, function reusability is reduced because the function may depend on the global variable being in a certain state.
  - Within a file, global variables are initialized in the order they are declared. This can lead to undefined behavior if one global variable depends on another global variable that has not yet been initialized.
  - **Static initialization order fiasco** is a problem that occurs when a global variable is initialized in one translation unit and is used in another translation unit before it has been initialized. This happens because the order of initialization of global variables across translation units is ambiguous and can lead to undefined behavior.

- If at all global variables are needed, it is best to use them inside a namespace and encapsulate them making sure they can be accessed directly only within the file (via static or const) and external access is only possible via functions.

- **Inline expansion** is a technique used by the compiler to replace a function call with the actual code of the function. This can improve performance by eliminating the overhead of a function call. Modern optimizing compilers make a decision on whether a function should be expanded inline or not. Some function calls **cannot be expanded** such as those defined in another translation unit.

- Historically, the `inline` keyword was used to hint to the compiler that a function should be expanded inline. However, this is no longer necessary because modern compilers are able to make this decision on their own.

- In mordern C++, the `inline` keyword is used to indicate that a function can be defined in multiple translation units without violating the One Definition Rule (ODR). `inline` functions have the following requirements:
  - The compiler needs to see the function definition in every translation unit that uses it. Only one such definition is allowed per translation unit or it results in a compiler error.
  - The definition can be after a function call if a forward declaration is provided. But, in this case, the compiler will likely not perform inline expansion.
  - The definition must be the same in all translation units or undefined behavior will occur.
 
- `inline` functions are typically defined in header files, which are included in multiple translation units. This is particluarly useful for small functions and **header-only libraries**. The following functions are implicitly inline:
  - functions defined inside a class, struct, or union type definition.
  - constexpr/consteval functions
  - functions implicitly instantiated from templates

- `inline` variables were introduced in C++17. They are similar to `inline` functions, but they can be used to define variables that can be shared across multiple translation units without violating the One Definition Rule (ODR). They have the same requirements as `inline` functions i.e., the compiler must be able to see identical and the full definition of the variable in every translation unit that uses it.

- Inline variables have external linkage by default so they are visible to the linker and can be de-duplicated.
  
- **static** has different meanings in different contexts:
  - A global variable has **static duration** meaning it is created when the program starts and destroyed when the program ends. 
  - `static` keyword when used with a global variable or function, it gives it **internal linkage**. 
  - `static` keyword when used with a local variable, it gives it **static duration**. This means that the variable is created when the program starts and destroyed when the program ends, but it is only visible within the block in which it is declared. 

- `static` local variables are best used to avoid expensive local object initialization each time the function is called. 

- `static` and `extern` are called **storage class specifiers**. They are used to specify the storage duration and linkage of variables and functions.

- A **qualified name** is a name that includes an associated scope. For example, in `std::cout`, `cout` is qualified by namespace `std` namespace. An **unqualified name** is a name that does not include a scoping qualifier. For example, just `cout` is unqualified. 

- A **using declaration** allows us to use an unqualified name (with no scope) for a qualified name. For example, `using std::cout;` allows us to use `cout` instead of `std::cout`. 

- A **using directive** allows all identifiers in a namespace to be referenced without qualification. For example, `using namespace std;` allows us to use `cout`, `cin`, etc. without qualification.

- `using` statements shouldn't be used before `#include` statements because the behavior of using statements is dependent on what identifiers have already been introduced which makes them order dependent. This is also why we should avoid using statements in header files (even inside functions since other factors (like previous #include statements) might change the way our using statement behaves).

- All content declared inside **unnamed namespaces** is treated as part of the parent namespace. However, identifiers declared inside unnamed namespaces are treated as having **internal linkage**.

- An `inline` namespace it typically used for versioning content. Like unnamed namespaces, inline namespaces are also treated as part of the parent namespace. However, inline namespaces do not affect linkage. eg:
  ```cpp
  #include <iostream>

  inline namespace V1 // declare an inline namespace named V1
  {
      void doSomething()
      {
          std::cout << "V1\n";
      }
  }

  namespace V2 // declare a normal namespace named V2
  {
      void doSomething()
      {
          std::cout << "V2\n";
      }
  }

  int main()
  {
      V1::doSomething(); // calls the V1 version of doSomething()
      V2::doSomething(); // calls the V2 version of doSomething()

      doSomething(); // calls the inline version of doSomething() (which is V1)

      return 0;
  }
  ```
- inline and unnamed namespaces can be used together but in such cases it is better to nest the anonymous namespace inside the inline namespace as it gives us an explicit namespace name we can use. Also, we can seperate out the parts for which we want internal linkage into the nested anonymous namespace from the ones whose linkage we don't care about.

- `case` labels in `switch` statements do not introduce a new scope and they are part of the scope of the switch block itself. This is why we need statements like `break` or `return` to exit the switch block or else the program will continue executing the next case label (ignoring the condition in that label). This is called **fall through** behavior.

- When we want intentional fall through behavior, we can use the `[[fallthrough]]` attribute (**Attributes** are a way to provide additional information to the compiler about the code).

- Because `switch` statements do not introduce a new scope, we can declare or define variables inside a case label. Initialization of variables in case labels is not allowed (except in the last case) because it is not guaranteed that the variable will be initialized before it is used as the switch could jump over the initialization statement.

- switch case labels can also be placed in a sequence. For example:
  ```cpp
  switch (x)
  {
      case 1:
      case 2:
      case 3:
          std::cout << "x is 1, 2, or 3\n";
          break;
      default:
          std::cout << "x is not 1, 2, or 3\n";
          break;
  }
  ```
  This is not fallthrough behavior.

- `goto` is an unconditional jump statement that can be used to jump to a labeled statement in the same function. It has **function scope** and can be used to jump to any label in the same function. It is generally considered bad practice to use `goto` because it can make code difficult to read and maintain and leads to a condition called **spaghetti code**.

- When jumping forward, we cannot jump over a variable declaration or initialization that is still in scope at the location being jumped to. This is because the variable will not be initialized before it is used, leading to undefined behavior. But, we can jump backwards over a variable declaration or initialization and it will be re-initialized when the program jumps back to that location. 

- `goto` statements are best used for error handling and moving out of deeply nested loops.

- `std::exit()` is a halt statement that resides in the `cstdlib` header file. It is used to **terminate a program normally** immediately. Since, `std::exit` doesn't cleanup any local variables in the current function or up the call stack, C++ provides us a function `std::atexit()` which allows us to call a function which is automatically called when `std::exit()` is called. This is useful for cleanup tasks like closing files, releasing memory and database connections, etc.

- In multithreaded programs, `std::exit()` can cause programs to crash because the thread calling `std::exit()`will cleanup static objects that other threads may still be using. `std::quick_exit()` terminates the program without cleaning up any static objects and may or may not do other types of cleanup. `std::at_quick_exit()` is similar to `std::atexit()` but it registers a function to be called when `std::quick_exit()` is called.

- `std::abort()` function causes the program to **terminate abnormally**. It does not do any cleanup.

- `std::terminate()` function is called implicitly when an exception is not handled. It does not do any cleanup and calls `std::abort()`.

- A pseudo random number generator (PRNG) is an algorithm that generates a sequence of numbers that approximates the properties of random numbers. PRNGs are not truly random because they use a deterministic algorithm to generate the numbers. However, they can produce a sequence of numbers that appears to be random for most practical purposes. eg:
```cpp
unsigned int LCG16() // our PRNG
{
    static unsigned int s_state{ 0 }; // only initialized the first time this function is called

    // Generate the next number

    // We modify the state using large constants and intentional overflow to make it hard
    // for someone to casually determine what the next number in the sequence will be.

    s_state = 8253729 * s_state + 2396403; // first we modify the state
    return s_state % 32768; // then we use the new state to generate the next number in the sequence
}
```

- An assertion is a statement that checks if a condition is true. If the condition is false, the program will terminate and print an error message. In C++, runtime assertions are implemented using the `assert()` macro. Assertions can be used to check for **preconditions** (conditions that must be true before a function is called), **postconditions** (conditions that must be true after a function is called), and **invariants** (conditions that must be true at all times). They can be disabled in release builds by defining the `NDEBUG` macro. 

- `assert()` is a macro that is defined in the `cassert` header file. `static_assert()` is a compile-time assertion that is a keyword in C++. It is used to check conditions at compile time. If the condition is false, the compiler will generate an error message. `static_assert()` is an assertion that is evaluated at compile time. Its expression must be a constant expression.

- A **value preserving conversion** (or **safe conversion**) is a type of conversion where every source value can be represented by the destination type without loss of information. Numeric promotions are value preserving conversions and hence safe. So, the compiler does not issue a warning for them.

- Numeric conversions can fall into 3 safety categories:
  - **Value preserving conversions**: Safe numeric conversions where the destination type can represent every source value without loss of information. 
  - **Reinterpretive conversions**: Unsafe numeric conversions where the converted value may be different from the original value but no data is lost. Signed/Unsigned conversions fall into this category.
  - **Lossy conversions**: Unsafe numeric conversions where data may be lost during conversion.

- A **narrowing conversion** is a potentially unsafe conversion where the destination type may not be able to represent all the values of the source type. For example, converting a `double` to an `int` is a narrowing conversion because the `int` type cannot represent all the values of the `double` type. Narrowing conversions should be avoided as they can be unsafe and a source or errors. In cases where narrowing conversion is unavoidabvle, it is best to use explicit casts to indicate that the conversion is intentional.

- A **usual arithmetic conversion** is a set of rules that the compiler uses to implicitly convert operands to the same type when they are used with an operator that requires them to be of the same type. Overloaded operators are not subject to usual arithmetic conversions rules.

- C++ supports 5 different types of casts:
  - `static_cast`: Performs compile time type conversions between compatible types.
  - `dynamic_cast`: Performs runtimetype conversions on pointers and references in a polymorphic hierarchy. 
  - `const_cast`: Used to add or remove the `const` qualifier from a variable.
  - `reinterpret_cast`: Reinterprets the bit-level representation of one type as if it were another type.
  - `C-style cast`: A combination of `static_cast`, `const_cast`, and `reinterpret_cast`. It is not type-safe and should be avoided in favor of the C++ casts.

- `using` keyword creates an alias for an existing data type. eg: `using Distance = double;` creates an alias `Distance` for the type `double`.

- `typedef` keyword is similar to `using` keyword but it is an older C-style way of creating type aliases. It is not recommended to use `typedef` in modern C++ code. 

- `auto` keyword has a number of uses. It can used for **type deduction** where the compiler deduces the type of a variable from its initializer. Type deduction drops constness and reference qualifiers.

- `auto` keyword can also be used as a function return type to have the compiler infer the return type of the function. This is called **trailing return type**. This is normally avoided.

- `auto` keyword when used with function parameters, doesn't invoke type deduction but rather a different feature called **function templates**.

- `decltype` keyword is used to query the type of an expression without evaluating it. It can be used to create type aliases and to deduce the type of a variable from its initializer.

- C++ performs **name mangling** of overloaded functions to create unique names for each function. So, even though the source code may look the same, the compiler generates different names for each function. Name mangling is compiler specific.

- In case we have a function that we explicitly don't want to be called, we can use the `= delete` syntax. `= delete` participates in all stages of function overload resolution so in case of ambiguous overloads, the compiler will throw an error. 

- C++ does not (as of C++23) support a function call syntax such as `print(,,3)` as a way to provide an explicit value. This has three major consequences:
  - In a function call, any explicitly provided arguments must be the leftmost arguments.
  - If a parameter is given a default argument, all subsequent parameters (to the right) must also be given default arguments.
  - If more than one parameter has a default argument, the leftmost parameter should be the one most likely to be explicitly set by the user.

- Default arguments can be included in function declarations or function definitions but not both. It is best to include them in the function declaration (given there is one) so that they are visible to all translation units that use the function. 

- **C++ templates** were designed to simplify the process of creating generic code. They allow us to write code that can work with any data type without having to write separate functions or classes for each data type. In a template, we can use one or more placeholder types which is a data type that is not known until the template is instantiated. The compiler generates the code for each data type when the template is instantiated. This is called **template instantiation**. Templates can work with types that don't even exist when the template is defined.

- A **function template** is a function-like definition that is used to generate one or more overloaded functions. The initial template used to generate the function is called the **primary template**. The generated functions are called **instantiated functions (or specializations)**. When creating a primary template, we use **placeholder types** (also called **type template parameters**) for any parameter types, return types, and types used in the function body that we want to be generic.

- C++ supports 3 types of template parameters:
  - **Type template parameters**: These are used to create generic functions and classes that can work with any data type. They are defined using the `template<typename T>` syntax.
  - **Non-type template parameters**: These are used to create generic functions and classes that can work with any value(or constexpr) of a specific type. They are defined using the `template<T value>` syntax.
  - **Template template parameters**: These are used to create generic functions and classes that can work with other templates. They are defined using the `template<template<typename> class T>` syntax.

- A `static` local variable used inside a function template, each instantiation of the function template will have its own copy of the static variable.

- A constexpr function is a function that is allowed to be called in a constant expression. To make a function a constexpr function, we simply use the constexpr keyword in front of the return type. Constexpr functions are only guaranteed to be evaluated at compile-time when used in a context that requires a constant expression. Otherwise they may be evaluated at compile-time (if eligible) or runtime. Constexpr functions are implicitly inline, and the compiler must see the full definition of the constexpr function to call it at compile-time.

- A `consteval` function is a function that must evaluate at compile-time. Consteval functions otherwise follow the same rules as constexpr functions. 

- `constexpr` functions can call non `constexpr` functions, but only when they are being used in a non-constant expression context. In this case, the non-constexpr function will be evaluated at runtime.

- `std::is_constant_evaluated` is a function that can be used to check if a function is being evaluated at compile-time or runtime. It returns true if the function is being evaluated at compile-time and false otherwise. However, it returns `false` even when it is being evaluated at compile time in a context where it is not required to be a constant expression. So, it is more of a check of whether **the function is being forced to be evaluated at compile-time** rather than if it is being evaluated at compile-time. Also, C++ 23 introduced `if consteval` which works in the same way but is more concise and cleaner.

- C++ supported compound data types:
  - Functions
  - C-style Arrays
  - Pointer types:
    - Pointer to object
    - Pointer to function
    - Pointer to member types:
    - Pointer to data member
    - Pointer to member function
  - Reference types:
    - L-value references
    - R-value references
  - Enumerated types:
    - Unscoped enumerations
    - Scoped enumerations
  - Class types:
    - Structs
    - Classes
    - Unions

- **Lvalue expressions** are those that evaluate to functions or identifiable objects (including variables) that persist beyond the end of the expression.

- **Rvalue expressions** are those that evaluate to values, including literals and temporary objects that do not persist beyond the end of the expression.

- References are **not objects** and do not have a memory address. They are simply an alias for an existing object. This means that **references cannot be reassigned** to refer to a different object after they are initialized. They are also **not pointers** and do not have pointer semantics.

- Trying to reassign a reference will change the value of the object it refers to. For example:
```cpp
#include <iostream>

int main()
{
    int x { 5 };
    int y { 6 };

    int& ref { x }; // ref is now an alias for x

    ref = y; // assigns 6 (the value of y) to x (the object being referenced by ref)
    // The above line does NOT change ref into a reference to variable y!

    std::cout << x << '\n'; // user is expecting this to print 5 but it prints 6

    return 0;
}
```

- References and **referents** (i.e., the objects they refer to) have independent lifetimes. This means that a reference can outlive the object it refers to, leading to undefined behavior if the reference is used after the object has been destroyed. This is called **dangling reference**.

- Since, references are not objects, there cannot be a reference to a reference. Hoowever, when we write something like:
```cpp
int var{};
int& ref1{ var };  // an lvalue reference bound to var
int& ref2{ ref1 }; // an lvalue reference bound to var
``` 
then `ref2` is not a reference to `ref1`, but rather a reference to the object that `ref1` refers to. This is because `ref1` is an lvalue expression that evaluates to the object it refers to. So, `ref2` is just another name for `var`.

- A reference to a reference (to an `int`) would have syntax `int&&` -- but since C++ doesn’t support references to references, this syntax was repurposed in C++11 to indicate an rvalue reference.

- Non-const lvalue references can bind to modifiable lvalues, while const lvalue references can bind to both modifiable and non-modifiable lvalues. In the second case, the reference is treated as a const lvalue reference and the object it refers to cannot be modified through the reference. However, the object itself can still be modified directly.

- Const lvalue references can bind to rvalues as well. If we try to bind a const lvalue reference to a value of a different type, the compiler will **create a temporary object** of the same type as the reference, initialize it using the value, and then bind the reference to the temporary. In such cases, the **lifetime of the temporary object is extended** to the lifetime of the reference. Also, nn such cases, modifying the original value will not reflect in the temporary object and hence the reference will **not be able to modify the original value**.

- `constexpr` references can bind only to static variables because (locals or globals) because the compiler knows where static objects will be instantiated in memory, so it can treat that address as a compile-time constant. This is not the case for local variables, which are created at runtime and hence their addresses are not known at compile-time.

- **Pass by reference** allows us to pass arguments to a function without making copies of those arguments each time the function is called.

- Because a reference to a non-const value can only bind to a modifiable lvalue (essentially a non-const variable), this means that pass by reference only works with arguments that are modifiable lvalues. n practical terms, this significantly limits the usefulness of pass by reference to non-const, as it means we can not pass const variables or literals. An easy way around this is to **pass by const reference**.

- Passing by const reference offers the same primary benefit as pass by non-const reference (avoiding making a copy of the argument), while also guaranteeing that the **function can not change** the value being referenced.
As a rule of thumb, pass fundamental types by value and class types by const reference.

- Despite the benefits of passing by reference, we don't use it everytime because:
  - For objects that are cheap to copy, the cost of copying is similar to the cost of binding, but accessing the objects is faster and the compiler is likely to be able to optimize better.
  - For objects that are expensive to copy, the cost of the copy dominates other performance considerations.

- Between, `std::string_view` and `const std::string&`, `std::string_view` is preferred because it is a lightweight, non-owning reference to a string that can be used to avoid unnecessary copies. It is also more efficient than `const std::string&` because it does not require the overhead of creating a temporary object.

- A **pointer** is an object that holds a memory address (typically of another variable) as its value. This allows us to store the address of some other object to use later. A type that specifies a pointer (e.g. `int*`) is called a pointer type. Much like reference types are declared using an ampersand (`&`) character, pointer types are declared using an asterisk (`*`)

- The ampersand (`&`) operator is used to get the address of an object, while the asterisk (`*`) operator is used to dereference a pointer (i.e., access the object that the pointer points to). `*ptr` will return an **lvalue expression** that refers to the object that `ptr` points to. 

- A pointer that has not been initialized to point to a valid object is called a **wild pointer**. Wild pointers contain garbage values and dereferencing them leads to undefined behavior. 

- We can use assignment with pointers in two different ways:
  - To change what the pointer is pointing at (by assigning the pointer a new address)
  - To change the value being pointed at (by assigning the dereferenced pointer a new value)

- The size of the pointer depends on the architecture of the machine for which the program is compiled. For example, on a 32-bit executable uses 32 bits (4 bytes) for pointers, while a 64-bit executable uses 64 bits (8 bytes) for pointers. The size of the pointer type is independent of the size of the object it points to because pointer just stores the memory address. For example, a pointer to an `int` is the same size as a pointer to a `double`, even though `int` and `double` are different sizes.

- A **dangling pointer** is a pointer that points to an object that has been destroyed or deallocated. Dereferencing a dangling pointer leads to undefined behavior.

- A pointer that doesn't point to anything is called a **null pointer**. Dereferencing a null pointer leads to undefined behavior. A null pointer is typically represented by the value `nullptr` in modern C++. In older C++ code, `NULL` or `0` was used to represent a null pointer.

- A null pointer implicitly converts to a boolean value of `false` and any other pointer value implicitly converts to a boolean value of `true`. We can use this to test if a pointer is null or not. However, conditionals can only identify if a pointer is null or not, not if it is a dangling pointer.

- Pointers have the additional abilities of being able to change what they are pointing at, and to be pointed at null. However, these pointer abilities are also inherently dangerous: A null pointer runs the risk of being dereferenced, and the ability to change what a pointer is pointing at can make creating dangling pointers easier. Since references can’t be bound to null, we don’t have to worry about null references. And because references must be bound to a valid object upon creation and then can not be reseated, dangling references are harder to create. Because they are safer, references should be favored over pointers, unless the additional capabilities provided by pointers are required.

- A **pointer to a const** object can be declared using the `const int*` syntax. Similar to references, a pointer to a `const` object can be used to bind to a const as well as a non-const object. However, the pointer itself can be changed to point to a different object. This is because the pointer is not const, only the object it points to is const.

- A **const pointer** can be declared using the `int* const` syntax. A const pointer can only be initialized once and cannot be changed to point to a different object. However, the object it points to can be modified. This is because the pointer itself is const, only the object it points to is not const.

- There are some rules we need to remember with regards to pointers, const pointers and pointers to const:
  - A non-const pointer (e.g. `int* ptr`) can be assigned another address to change what it is pointing at.
  - A const pointer (e.g. `int* const ptr`) always points to the same address, and this address can not be changed.
  - A pointer to a non-const value (e.g. `int* ptr`) can change the value it is pointing to. These can not point to a const value.
  - A pointer to a const value (e.g. `const int* ptr`) treats the value as const when accessed through the pointer, and thus can not change the value it is pointing to. These can be pointed to const or non-const l-values (but not r-values, which don’t have an address).
  - A const before the asterisk (e.g. const int* ptr) is associated with the type being pointed to. Therefore, this is a pointer to a const value, and the value cannot be modified through the pointer.
  - A const after the asterisk (e.g. int* const ptr) is associated with the pointer itself. Therefore, this pointer cannot be assigned a new address.

- Apart from pass by value and pass by reference, C++ also supports **pass by address**. This is done by passing a pointer to the function instead of the actual object. This allows us to modify the original object without making a copy of it. Pass by address is similar to pass by reference, but it is less safe because it allows us to pass null pointers and dangling pointers. It is also clutters the code with pointer syntax. Hence, pass by reference is preferred over pass by address. 

- We can also pass by address using a reference to a pointer using the syntax `int*& ptr`. This allows us to modify the pointer itself (i.e., change what it is pointing to) as well as the object it points to. Pass by referebces to pointers are not commonly used and the syntax can be easily messed up by using `int&* ptr` instead of `int*& ptr`. however, this will result in a compiler error because a pointer to a reference cannot exist as a reference is not an object.

- The reason why `0` or `NULL` is not preferred is because 0 can be interpreted as an integer literal, and NULL is a macro that is not defined by the C++ standard (it could be `0`,`0L`, or `((void*)0)`). This can lead to confusion and ambiguity in the code. eg:  In the case of two overloaded functions `void foo(int)` and `void foo(int*)`, calling `foo(0)` will call the first function because `0` is an integer literal. If that was not the intended function, then we may have a problem. On the other hand, `foo(NULL)` could call either function depending on how NULL is defined.

- `nullptr` is the only value that `std::nullptr_t` can take. It is a null pointer constant that can be used to represent a null pointer in a type-safe way. It is not an integer literal and does not have any other meaning in C++. If we want to write a function that accepts only a `nullptr` literal argument, we can make the parameter a `std::nullptr_t`. If we have an overloaded function that accepts a pointer and we set a pointer to `nullptr` and call the function, it will call the pointer overload because C++ matches overloads on type not values and a pointer (with value of `nullptr`) cannot implicitly convert to `std::nullptr_t`.

- Pass by address just copies an address from the caller to the called function -- which is just passing an address by value. Therefore, we can conclude that C++ really passes everything by value! The properties of pass by address (and reference) come solely from the fact that we can dereference the passed address to change the argument, which we can not do with a normal value parameter!

- Similar to pass by reference (or address), we can also return by reference (or address). However, there is one major caveat: Objects returned by reference must live beyond the scope of the function returning the reference, or a dangling reference will result. Never return a (non-static) local variable or temporary by reference.

- Reference lifetime extension for temporary objects does not work across function boundaries.

- Avoid returning references to non-const local static variables.

- Prefer return by reference over return by address unless the ability to return “no object” (using `nullptr`) is important.

- Parameters used only for receiving input from the caller are called **in parameters**. Parameters used only for returning information to the caller are called **out parameters**. Parameters used for both input and output are called **in-out parameters**.

- A **top-level const** is a const that applies to the object itself. For example, `const int x`, `int* const ptr`. There is no top-level const syntax for references because references are implicitly const. A **low-level const** is a const that applies to the object being pointed to. For example, `const int* ptr`, `const int& ref`. A reference to a const object is a low-level const. A pointer can have top-level, low-level, or both types of const. eg: `const int* const ptr`

- Type deduction only drops top-level constness.
```cpp
#include <string>

const std::string& getConstRef(); // some function that returns a const reference

int main()
{
    auto ref1{ getConstRef() };        // std::string (reference and top-level const dropped)
    const auto ref2{ getConstRef() };  // const std::string (reference dropped, const dropped, const reapplied)

    auto& ref3{ getConstRef() };       // const std::string& (reference dropped and reapplied, low-level const not dropped)
    const auto& ref4{ getConstRef() }; // const std::string& (reference dropped and reapplied, low-level const not dropped)

    return 0;
}
```
In the above example, since `getConstRef()` returns a `const std::string&`, the reference is dropped first (for initialization), leaving us with a `const std::string`. This const is now a top-level const, so it is also dropped, leaving the deduced type as `std::string` for `ref1`. For `ref2`, this is similar to the `ref1` case, except we’re reapplying the `const` qualifier, so the deduced type is `const std::string`. Things get more interesting with `ref3`. Normally the reference would be dropped first, but since we’ve reapplied the reference, it is not dropped. That means the type is still `const std::string&`. And since this const is a low-level const, it is not dropped. Thus the deduced type is `const std::string&`. The `ref4` case works similarly to `ref3`, except we’ve reapplied the const qualifier as well. Since the type is already deduced as a reference to `const`, us reapplying `const` here is redundant. That said, using `const` here makes it explicitly clear that our result will be const (whereas in the `ref3` case, the constness of the result is implicit and not obvious).

- Constexpr is not part of an expression’s type, so it is not deduced by `auto`.

- Type deduction doesn't drop pointers.
```cpp
#include <string>

std::string* getPtr(); // some function that returns a pointer

int main()
{
    auto ptr1{ getPtr() };  // std::string*
    auto* ptr2{ getPtr() }; // std::string*

    return 0;
}
```
Here, when we use `auto` the type deduced is a pointer `std:string*` whereas when we use `auto*` the type deduced is `std::string`. However, for `auto*` the pointer is reapplied later and the practical effect is the same as `ptr1` and `ptr2` are both pointers to `std::string`. However, for `auto*`, the initialzer must be a pointer type. eg: `auto* ptr4{ *getPtr() }` will not compile because the initializer is not a pointer type.  

- Just like with references, only top-level constness is dropped when using during pointer type deduction.

- `std::optional` is a class template that represents an object that may or may not contain a value. It can be used in the followiing ways:
```cpp
std::optional<int> o1 { 5 };            // initialize with a value
std::optional<int> o2 {};               // initialize with no value
std::optional<int> o3 { std::nullopt }; // initialize with no value

if (o1.has_value()) // call has_value() to check if o1 has a value
if (o2)             // use implicit conversion to bool to check if o2 has a value

std::cout << *o1;             // dereference to get value stored in o1 (undefined behavior if o1 does not have a value)
std::cout << o2.value();      // call value() to get value stored in o2 (throws std::bad_optional_access exception if o2 does not have a value)
std::cout << o3.value_or(42); // call value_or() to get value stored in o3 (or value `42` if o3 doesn't have a value)
```

- Even though we can dereference it to get the value, `std::optional` has value semantics (meaning it contains a value) unlike pointers which have reference semantics.

- C++ allows us to create new, custom types that we can use in our programs. These types are called **user-defined types** or **program-defined types**. C++ has 2 different categories of compound types that can be used to create program-defined types:
  - **Enumerated types**: These are used to create a new type that can take on a limited set of values. They are defined using the `enum` keyword. Enumerated types can be either scoped or unscoped.
  - **Class types**: These are used to create a new type that can have data members and member functions. They are defined using the `class` keyword. Class types can be struct, class, or union types.

- A program-defined type must have a name and a definition before it can be used. This is called **type definition**. Type definitions end with a semicolon.

- Type definitions that need to be used in multiple translation units should be placed in a header file and included in the translation units that need to use them. Type definitions are **partially exempt from the one definition rule (ODR)**. They can be defined in multiple translation units, but they must be identical in all translation units.

- C++ defines the term  **user-defined type** as any class type or enumerated type that is defined by you, the standard library, or the implementation (e.g. types defined by the compiler to support language extensions). Perhaps counter-intuitively, this means `std::string` (a class type defined in the standard library) is considered to be a user-defined type! To provide additional differentiation, the C++20 language standard helpfully defines the term **program-defined type** to mean class types and enumerated types that are not defined as part of the standard library, implementation, or core language. In other words, “program-defined types” only include class types and enum types that are defined by us (or a third-party library).

- An **enumeration** (also called an **enumerated type** or an **enum**) is a compound data type whose values are restricted to a set of named symbolic constants (called **enumerators**). The initializer for an enumerated type must be one of the defined enumerators for that type. eg:
```cpp
// Define a new unscoped enumeration named Color
enum Color
{
    // Here are the enumerators
    // These symbolic constants define all the possible values this type can hold
    // Each enumerator is separated by a comma, not a semicolon
    red,
    green,
    blue, // trailing comma optional but recommended
}; // the enum definition must end with a semicolon

int main()
{
    // Define a few variables of enumerated type Color
    Color apple { red };   // my apple is red
    Color shirt { green }; // my shirt is green
    Color cup { blue };    // my cup is blue

    Color socks { white }; // error: white is not an enumerator of Color
    Color hat { 2 };       // error: 2 is not an enumerator of Color

    return 0;
}
```

- Unscoped enumerations put the enumerators in the same scope as the enumeration itself. The enumerators of an enumeration defined within a function will shadow identically named enumerators defined in the global scope.

- When we define an enumeration, each enumerator is automatically associated with an integer value based on its position in the enumerator list. By default, the first enumerator is given the integral value `0`, and each subsequent enumerator has a value one greater than the previous enumerator:
```cpp
enum Color
{
    black,   // 0
    red,     // 1
    blue,    // 2
    green,   // 3
    white,   // 4
    cyan,    // 5
    yellow,  // 6
    magenta, // 7
};
```

- Enumerators can be explicitly assigned values. If we assign a value to an enumerator, the next enumerator will have a value one greater than the previous enumerator unless we explicitly assign it a value as well. For example:
```cpp
enum Animal
{
    cat = -3,    // values can be negative
    dog,         // -2
    pig,         // -1
    horse = 5,
    giraffe = 5, // shares same value as horse
    chicken,     // 6
};
```

- If an enumeration is zero-initialized, it will be given a value of `0` even if there is no enumerator with that value. Make the enumerator representing `0` the one that is the best default meaning for your enumeration. If no good default meaning exists, consider adding an “invalid” or “unknown” enumerator that has value `0`, so that state is explicitly documented and can be explicitly handled where appropriate.

- C+++ doesn't specify which **underling type** to be used for enumerated types. We can also change the underlying type of an enumerated type. eg: `enum Color : std::int8_t { red, green, blue };`

- Integral values cannot be implicitly converted to enumerated types. We can use explicit conversion using `static_cast` to convert an integral value to an enumerated type. It is also safe to static_cast any integral value that is in range of the target enumeration’s underlying type, even if there are no enumerators representing that value. Static casting a value outside the range of the underlying type will result in undefined behavior.

- **Operator overloading** lets us define overloads of existing operators, so that we can make those operators work with our program-defined data types. To do this, we define a function that has the same name as the operator we want to overload. eg: `operator+(x, y)`. We can use operator overloading to print the string values of our enumerated types using the enumerator names. The same can bbe done to input string values to our enumerated types.

- **Scoped enumerations** (also called **strongly typed enumerations**) are a new feature introduced in C++11. They are defined using the `enum class` or `enum struct` keyword. Scoped enumerations are similar to unscoped enumerations, but they have a few key differences:
  - The enumerators of a scoped enumeration are not put in the same scope as the enumeration itself. This means that we must use the enumeration name to access the enumerators. For example: `Color::red`.
  - The enumerators of a scoped enumeration do not implicitly convert to integral types. This means that we cannot assign an integral value to a scoped enumeration variable without an explicit cast.

- In cases where it is useful to have the enumerators of a scoped enumeration be implicitly convertible to integral types, we can use `static_cast` or `std::to_underlying` (in the utility header) to convert the enumerator to convert it to its underlying type.

- `using enum` statement imports all of the enumerators from an enum into the current scope. When used with an enum class type, this allows us to access the enum class enumerators without having to prefix each with the name of the enum class.