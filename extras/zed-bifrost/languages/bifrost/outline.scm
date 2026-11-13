; `module helper = { ... }`
(module
  "module" @context
  name: (identifier) @name) @item

; `let main = [deps] (args) => T ...`, `let Box[T] = struct { ... }`, `let x = ...`:
; a top-level (or module) definition, with a function's parameters and result.
(assignment
  "let" @context
  (identifier) @name
  (type_parameters)? @context
  (function_definition
    (parameter_list) @context
    "=>" @context
    .
    (type_or_object) @context)?) @item

; A struct's or union's fields and functions: `let hits: i64`, `static let new = ...`
(struct_field
  "static"? @context
  "let" @context
  (identifier) @name
  (local_function_definition
    (parameter_list) @context
    "=>" @context
    (type_or_object) @context)?) @item
