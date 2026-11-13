; Functions: `let f = [deps] (args) => T { ... }`, a struct's functions, lambdas
(assignment
  (function_definition
    (block_expression
      "{"
      (_)* @function.inside
      "}"))) @function.around

(assignment
  (function_definition
    (expression) @function.inside)) @function.around

(struct_field
  (local_function_definition
    (block_expression
      "{"
      (_)* @function.inside
      "}"))) @function.around

(struct_field
  (local_function_definition
    (expression) @function.inside)) @function.around

(local_assignment
  (expression
    (local_function_definition
      (block_expression
        "{"
        (_)* @function.inside
        "}")))) @function.around

(local_assignment
  (expression
    (local_function_definition
      (expression) @function.inside))) @function.around

(local_function_definition
  (block_expression
    "{"
    (_)* @function.inside
    "}")) @function.around

(local_function_definition
  (expression) @function.inside) @function.around

; Objects, unions and modules. Their fields are comma-separated, and `(_)*`
; alone stops at the first comma, so the commas are matched too.
(assignment
  (struct_assignment
    "{"
    [
      (_)
      ","
    ]* @class.inside
    "}")) @class.around

(assignment
  (union_definition
    "{"
    [
      (_)
      ","
    ]* @class.inside
    "}")) @class.around

(assignment
  (union_definition)) @class.around

(module
  "{"
  (_)* @class.inside
  "}") @class.around

(comment)+ @comment.around
