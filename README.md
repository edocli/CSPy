# CSPy - AP CSP Pseudo-code Interpreter

A pure Python interpreter for the **AP Computer Science Principles (AP CSP)** 2026 exam reference pseudo-code. It fully implements the standard syntax, including 1-based list indexing, procedures, recursion, and the standard library functions.

[AP CSP Exam Reference Sheet](https://apcentral.collegeboard.org/media/pdf/ap-computer-science-principles-exam-reference-sheet.pdf)

> Note: This is just a final project for my university dual-credit course. The code is total spaghetti and it severely lacks error handling. I might update it later, but probably won't.

~~Special thanks to Gemini3Pro for the invaluable assistance 🤓~~

## ✨ Features

* **Standard Syntax Compliance**: Supports variable assignment (`<-`), arithmetic operations, and logical comparisons.
* **Control Flow**: Full support for `IF/ELSE`, `REPEAT TIMES`, `REPEAT UNTIL`, and `FOR EACH` loops.
* **AP CSP Specific Behavior**:
  * **1-Based Indexing**: Lists are indexed starting from 1.
  * **Copy by Value**: List assignment (`a <- b`) creates an independent copy of the list, strictly adhering to the AP CSP exam mental model (modifying the new list does not affect the original).
* **Procedures & Recursion**: Supports `PROCEDURE` definitions with parameter passing, local scoping, and `RETURN` statements.
* **Built-in Library**: Includes `DISPLAY`, `INPUT`, `RANDOM`, `APPEND`, `INSERT`, `REMOVE`, and `LENGTH`.
* **Zero Dependencies**: Built entirely with the Python standard library (`dataclasses`, `enum`, `sys`, `os`, `random`).

## 🚀 Quick Start

### Requirements

* Python 3.10+ (Recommended to run via `uv` or standard python interpreter)

### Running the Interpreter

Assuming your interpreter script is named `cspy.py` and your AP CSP code is in `script.csp`.

1. **Using standard Python**:

    ```bash
    python cspy.py script.csp
    ```

2. **Using `uv`**:

    ```bash
    uv run cspy.py script.csp
    ```

## 📖 Syntax Guide

### Variables & Assignment

```text
x <- 10
str <- "Hello"
is_valid <- TRUE
```

### List Operations

> **Note**: Indexes start at **1**.

```text
# Definition
nums <- [10, 20, 30]

# Access & Modification
val <- nums[1]      # Gets the first element (10)
nums[2] <- 99       # Modifies the second element

# Built-in List Functions
APPEND(nums, 40)    # [10, 99, 30, 40]
INSERT(nums, 1, 5)  # [5, 10, 99, 30, 40]
REMOVE(nums, 2)     # Removes the element at index 2
len <- LENGTH(nums)
```

### Control Flow

```text
# Conditionals
IF (x > 5) {
    DISPLAY("Big")
} ELSE {
    DISPLAY("Small")
}

# Loops
REPEAT 3 TIMES {
    DISPLAY("Hello")
}

REPEAT UNTIL (x = 10) {
    x <- x + 1
}

# Iteration
FOR EACH item IN nums {
    DISPLAY(item)
}
```

### Procedures

```text
PROCEDURE add(a, b) {
    RETURN(a + b)
}

sum <- add(5, 10)
```

## 🛠️ Architecture

The interpreter follows a classic **Recursive Descent** architecture:

1. **Lexer**: Tokenizes the source code, handling composite symbols (e.g., `<=`, `<-`) and keywords.
2. **Parser**: Generates an **Abstract Syntax Tree (AST)** using Python `dataclasses` for a clean and memory-efficient structure.
3. **Interpreter**: Traverses the AST using the **Visitor Pattern**.
      * Maintains an `Environment Stack` to handle local variables and recursion depth.
      * Enforces strict type checks and AP CSP-specific memory behaviors (e.g., list copying).

## 📝 Example

**Recursive Fibonacci Sequence:**

```text
PROCEDURE fib(n) {
    IF (n <= 2) {
        RETURN(1)
    }
    RETURN(fib(n - 1) + fib(n - 2))
}

DISPLAY("Fibonacci of 10 is:")
result <- fib(10)
DISPLAY(result)
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Planned improvements include:

* Detailed error reporting (Line/Column numbers).
* Interactive REPL mode.

## 📄 License

Apache License 2.0 OR MIT License
