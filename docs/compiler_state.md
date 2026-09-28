# Compiler State Management

## Overview

The Bifrost compiler uses a centralized state management system based on a context manager pattern. This ensures that all global state is properly initialized, tracked, and cleaned up during compilation.

## The Problem

Previously, the compiler had global state scattered across multiple modules:

```python
# context.py
ctx = RAIIMLIRContext()  # Global MLIR context

# streams.py
_declared_iob_func: bool = False  # Global declaration tracker

# _types.py
_DECLARED_FUNCTIONS: set[str] = set()  # Global function tracker
```

**Issues with this approach:**
- Hard to test in isolation
- State leaks between compilations
- No way to reset state cleanly
- Potential race conditions with multithreading
- Unclear initialization order

## The Solution: `CompilerState`

All global state is now managed by the `CompilerState` class, accessed through the `compiler_context()` context manager:

```python
from bifrost.core.compiler_state import compiler_context

with compiler_context() as state:
    # state.mlir_context - The MLIR context
    # state.declared_functions - Set of declared functions
    # state.declared_iob_func - Whether __acrt_iob_func is declared
    
    # Your compilation code here
    compile_program()
```

## Architecture

```
┌─────────────────────────────────────┐
│     compiler_context()              │
│  (Context Manager)                  │
└──────────────┬──────────────────────┘
               │ creates & manages
               ▼
┌─────────────────────────────────────┐
│     CompilerState                   │
│  ┌───────────────────────────────┐  │
│  │ mlir_context: RAIIMLIRContext │  │
│  │ declared_functions: set[str]  │  │
│  │ declared_iob_func: bool       │  │
│  └───────────────────────────────┘  │
└──────────────┬──────────────────────┘
               │ accessed by
               ▼
┌─────────────────────────────────────┐
│   get_state() -> CompilerState      │
│  (Called by modules that need state)│
└─────────────────────────────────────┘
```

## Usage

### Basic Compilation

```python
from bifrost.core.compiler_state import compiler_context


def compile_program():
    with compiler_context() as state:
        # State is automatically initialized
        backend = LLVMJITBackend()
        module = ExplicitlyManagedModule()

        # Declare functions (they use get_state() internally)
        streams.declare_iob_func()
        io.declare_io_functions()

        # Define your program
        @func.func
        def main():
            # Your code
            pass

        # Compile
        main.emit()
        module.finish()

        # State is automatically cleaned up on exit
```

### Testing

The context manager makes testing much easier:

```python
def test_string_creation():
    with compiler_context() as state:
        # Fresh state for this test
        string = DynamicString("Hello, test!")
        assert string.value == "Hello, test!"

    # State is cleaned up automatically


def test_function_declaration():
    with compiler_context() as state:
        # Each test gets isolated state
        fputs = get_fputs()
        fputs.ensure_declared()

        assert "fputs" in state.declared_functions

    # State doesn't leak to other tests
```

## Internal Details

### How Modules Access State

Modules that need compiler state use `get_state()`:

```python
from bifrost.core.compiler_state import get_state


def declare_iob_func():
    state = get_state()
    if not state.declared_iob_func:
        # Declare the function
        llvm.func(...)
        state.declared_iob_func = True
```

### Lazy Initialization

Some objects (like function type constants) need the MLIR context at creation time. These use lazy initialization:

```python
# constants.py
_FPUTS: FunctionType | None = None


def get_fputs() -> FunctionType:
    global _FPUTS
    if _FPUTS is None:
        # Only created when first accessed (inside a compiler context)
        _FPUTS = FunctionType(
            name="fputs",
            attribute_str="!llvm.func<i32 (ptr, ptr)>",
            result=t.IntegerType.get_signless(32, get_context().context),
        )
    return _FPUTS


# Can be used as a function
FPUTS = get_fputs
```

### Error Handling

The context manager ensures proper cleanup even on errors:

```python
try:
    with compiler_context() as state:
        # If something goes wrong...
        raise CompilationError("Bad code!")
except CompilationError:
    # State is still cleaned up properly
    pass

# Next compilation gets fresh state
with compiler_context() as state:
    # No leftover state from previous failed compilation
    pass
```

## Benefits

### 1. **Testability**
Each test gets isolated state that doesn't affect other tests:
```python
def test_a():
    with compiler_context():
        declare_iob_func()  # OK


def test_b():
    with compiler_context():
        declare_iob_func()  # Also OK - fresh state
```

### 2. **No State Leaks**
State is automatically reset between compilations:
```python
compile_module_1()  # Declares functions
compile_module_2()  # Fresh state - must declare again
```

### 3. **Clear Lifecycle**
The context manager makes it obvious when the compiler is active:
```python
# Outside context - can't compile
with compiler_context():
    # Inside context - can compile
    pass
# Outside context - can't compile
```

### 4. **Thread Safety Foundation**
While not currently thread-safe, the centralized state makes it easier to add thread safety later (e.g., with thread-local storage).

### 5. **Explicit Dependencies**
Modules explicitly declare they need compiler state via `get_state()`, making dependencies clear.

## Migration Guide

If you're updating old code:

### Before
```python
from bifrost.core import context

# Context was a global singleton
mlir_ctx = context.ctx.context
```

### After
```python
from bifrost.core.compiler_state import compiler_context
from bifrost.core.context import get_context

with compiler_context():
    # Access context when needed
    mlir_ctx = get_context().context
```

### Function Constants

### Before
```python
from bifrost.standard.io.constants import FPUTS

# Used directly
FPUTS.ensure_declared()
```

### After
```python
from bifrost.standard.io.constants import FPUTS

# Call as a function
FPUTS().ensure_declared()
```

## Future Improvements

Potential enhancements to the state management system:

1. **Thread-local state** - Allow parallel compilation of multiple modules
2. **State snapshots** - Save/restore state for incremental compilation
3. **State validation** - Check that state is valid before compilation
4. **Diagnostic tracking** - Store compilation errors/warnings in state
5. **Compilation metrics** - Track performance data per compilation session

## Summary

The `CompilerState` context manager centralizes all global state, making the compiler:
- **More testable** - Clean state isolation between tests
- **More maintainable** - Clear state lifecycle and dependencies
- **More robust** - Automatic cleanup, even on errors
- **More extensible** - Foundation for future features like parallel compilation

Always wrap compilation in `compiler_context()`:

```python
with compiler_context() as state:
    # Your compilation here
    pass
```
