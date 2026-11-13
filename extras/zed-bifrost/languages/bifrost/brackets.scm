("(" @open
  ")" @close)

("[" @open
  "]" @close)

("{" @open
  "}" @close)

; `#[...]`, `#(...)`, `#{...}`: lists, tuples, records and record types
(list
  "#[" @open
  "]" @close)

(tuple
  "#(" @open
  ")" @close)

(record
  "#{" @open
  "}" @close)

(record_type
  "#{" @open
  "}" @close)

; `<i64, str>`: a tuple's type
(tuple_type
  "<" @open
  ">" @close)

((string
  "\"" @open
  "\"" @close)
  (#set! rainbow.exclude))
