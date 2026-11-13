; Bifrost highlights for Zed.
; In Zed a later pattern wins over an earlier one for the same node, so the
; general rules come first and the specific ones after them.

; ---------------------------------------------------------------------------
; Identifiers (fallbacks)
; ---------------------------------------------------------------------------

(identifier) @variable

; `a.b.c`: the value read from, then its fields.
(child_annotation
  .
  (simple_identifier) @variable)

(child_annotation
  "."
  (simple_identifier) @property)

; `struct { ... } & models.Person`
(base
  (simple_identifier) @variable)

(base
  (simple_identifier) @type
  .)

; `_` in a match arm or as a value
(default_var) @variable.special

; `this` and `super` (keywords in dependency lists, names in expressions)
[
  "this"
  "super"
] @variable.special

((simple_identifier) @variable.special
  (#any-of? @variable.special "this" "super"))

; ---------------------------------------------------------------------------
; Types
; ---------------------------------------------------------------------------

[
  "i8"
  "i16"
  "i32"
  "i64"
  "u8"
  "u16"
  "u32"
  "u64"
  "f32"
  "f64"
  "h8"
  "h16"
  "h32"
  "h64"
  "o8"
  "o16"
  "o32"
  "o64"
  "b8"
  "b16"
  "b32"
  "b64"
  "bool"
  "str"
] @type.builtin

(type_or_object
  (identifier) @type)

(type_or_object
  (child_annotation
    (simple_identifier) @type
    .))

(generic_type
  base: (identifier) @type)

(generic_type
  base: (child_annotation
    (simple_identifier) @type
    .))

; `let Box[T] = ...`
(type_parameters
  name: (identifier) @type)

; ---------------------------------------------------------------------------
; Fields, parameters and arguments
; ---------------------------------------------------------------------------

(parameter
  (identifier) @variable.parameter)

(named_argument
  name: (identifier) @variable.parameter)

(record_field
  name: (identifier) @property)

(record_type_field
  name: (identifier) @property)

; `let hits: i64` in a struct
(struct_field
  (identifier) @property)

; ---------------------------------------------------------------------------
; Functions
; ---------------------------------------------------------------------------

; Calls: `f(x)`, and methods: `a.f(x)`, `xs |> .sorted()`
(user_function_call
  function: (identifier) @function.call)

(child_annotation
  "."
  (function_call
    (user_function_call
      function: (identifier) @function.method.call)))

(pipe_method
  (function_call
    (user_function_call
      function: (identifier) @function.method.call)))

; `x |> f`, `x |> m.f`: a function's name piped into
(pipe_expression
  into: (expression
    (identifier) @function))

(pipe_expression
  into: (expression
    (child_annotation
      "."
      (simple_identifier) @function
      .)))

; Definitions: `let f = [deps] (...) => T ...`
(assignment
  (identifier) @function
  (function_definition))

(assignment
  (identifier) @function
  (expression
    (local_function_definition)))

(local_assignment
  (identifier) @function
  (expression
    (local_function_definition)))

; A struct's functions: `let to_string = [super] () => str ...`
(struct_field
  (identifier) @function.method
  (local_function_definition))

; The functions a function may call: `[stdio.printf, helper, this]`
(dependency_list
  (identifier) @function)

(dependency_list
  (child_annotation
    (simple_identifier) @function
    .))

(local_dependency_list
  (identifier) @function)

(local_dependency_list
  (child_annotation
    (simple_identifier) @function
    .))

; Builtins: `import(...)`, `len(xs)`, `panic(...)`
(builtin) @function.builtin

; ---------------------------------------------------------------------------
; Type and module definitions
; ---------------------------------------------------------------------------

; `let Name = struct { ... }`, `let Vehicle = Car | Airplane on "type"`
(assignment
  (identifier) @type
  [
    (struct_assignment)
    (union_definition)
  ])

(module
  name: (identifier) @module)

(export_statement
  module: (identifier) @module)

; ---------------------------------------------------------------------------
; Objects are PascalCase, constants SCREAMING_CASE
; ---------------------------------------------------------------------------

((identifier) @type
  (#match? @type "^[A-Z][a-zA-Z0-9]*$"))

((child_annotation
  (simple_identifier) @type)
  (#match? @type "^[A-Z][a-zA-Z0-9]*$"))

; Two or more capitals (a single one, `T`, is a type parameter): `SPACE`, `LEFT_SHIFT`
((identifier) @constant
  (#match? @constant "^[A-Z][A-Z0-9_]+$"))

((child_annotation
  "."
  (simple_identifier) @constant)
  (#match? @constant "^[A-Z][A-Z0-9_]+$"))

; Conversions: `f32(column)`, `i64(x)`
((user_function_call
  function: (identifier) @type.builtin)
  (#any-of? @type.builtin
    "i8" "i16" "i32" "i64" "u8" "u16" "u32" "u64" "f32" "f64"
    "h8" "h16" "h32" "h64" "o8" "o16" "o32" "o64" "b8" "b16" "b32" "b64"
    "bool" "str"))

; ---------------------------------------------------------------------------
; Literals
; ---------------------------------------------------------------------------

(number) @number

(number
  (float) @number.float)

(boolean) @boolean

(null) @constant.builtin

(comment) @comment

(string) @string

[
  (string_escape)
  (string_brace)
] @string.escape

; `"Hello, {name}!"`, `"{price:.2f}"`
(interpolation) @embedded

; (Its `{`, `}` and `:` are at the end of the file, after the brackets'.)

(format_spec) @string.special

; `let type: "employee"`: a field that only ever holds this text
(literal_type
  (string) @string.special)

; `on "type"`: the field telling a union's objects apart
(union_definition
  discriminator: (string) @string.special)

; `import("std:mem")`: the module's path
((builtin_call
  function: (builtin) @_import
  arguments: (expression
    (literal
      (string) @string.special.path)))
  (#eq? @_import "import"))

; ---------------------------------------------------------------------------
; Keywords
; ---------------------------------------------------------------------------

"let" @keyword

[
  "struct"
  "module"
] @keyword.type

"static" @keyword.modifier

"on" @keyword.operator

(export_statement
  "export" @keyword.export)

[
  "if"
  "else"
  "match"
] @keyword.conditional

[
  "while"
  "forall"
] @keyword.repeat

"in" @keyword.operator

"return" @keyword.return

[
  "async"
  "await"
] @keyword.coroutine

; ---------------------------------------------------------------------------
; Operators
; ---------------------------------------------------------------------------

[
  "="
  "=>"
  "<-"
  "->"
  "|>"
  "?"
  "!"
  "+"
  "-"
  "*"
  "**"
  "/"
  "%"
  "=="
  "!="
  "<"
  "<="
  ">"
  ">="
  "<<"
  ">>"
  "&"
  "&&"
  "|"
  "||"
  "^"
  (ellipsis)
] @operator

; ---------------------------------------------------------------------------
; Punctuation
; ---------------------------------------------------------------------------

[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
  "#["
  "#("
  "#{"
  "[]"
] @punctuation.bracket

; `<i64, str>`: a tuple's type
(tuple_type
  [
    "<"
    ">"
  ] @punctuation.bracket)

[
  ","
  "."
  ":"
] @punctuation.delimiter

; Interpolation braces again: they lex as the same `{` and `}` as blocks.
(interpolation
  "{" @punctuation.special
  "}" @punctuation.special)

(interpolation
  ":" @punctuation.special)
