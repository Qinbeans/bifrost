"""``bfc init <name>``: create a project, laid out the way the compiler expects.

```
name/
├── .gitignore
├── README.md
├── config.yaml          # package, flags, libraries, externs; entry: src/name/main.bif
└── src/
    └── name/
        └── main.bif     # the entry point
```

With ``src/``, imports resolve from it, as in Python: ``import("name.utils:text")``
is ``src/name/utils.bif``.
"""

from pathlib import Path

from bifrost.naming import is_snake_case


class ScaffoldError(ValueError):
    """The project cannot be created."""


GITIGNORE = """\
# Bifrost build output
build/
"""

CONFIG = """\
package:
  name: {name}
  version: 0.1.0
  description: {description}
  entry: src/{name}/main.bif
  # async lets main be async (`let main = [...] async () => ...`), run on the event loop.
  type: sync

flags:
  optimization: 2
  linker: clang

# Static libraries (paths, relative to this file) and system ones (names, like m).
libraries: []

# C declarations Bifrost code can import. Generate them from headers with:
#   bfc config traverse -i path/to/header.h
externs: []
"""

MAIN = """\
let stdio = import("std:stdio")
let fmt = import("std:fmt")

let main = [fmt.format, stdio.puts] () => null {{
    let greeting = fmt.format("Hello from %s!", "{name}")
    stdio.puts(greeting)
}}
"""

README = """\
# {name}

{description}

## Build and run

```bash
bfc build
./build/{name}
```

`bfc build` compiles `src/{name}/main.bif` (the `entry` in `config.yaml`).
Modules in `src/` import each other from there: `import("{name}.utils:text")`
is the module `text` of `src/{name}/utils.bif`.
"""


def create(name: str, parent: Path, description: str = "") -> Path:
    """Create the project ``name`` in ``parent``; return its folder.

    Raises:
        ScaffoldError: If ``name`` is not snake_case, or the folder already exists.

    """
    if not is_snake_case(name) or name.startswith("_") or name == "std":
        msg = f"'{name}' is not a project name: use snake_case, like my_api"
        raise ScaffoldError(msg)
    project = parent / name
    if project.exists():
        msg = f"{project} already exists"
        raise ScaffoldError(msg)
    description = description or f"The {name} project"
    files = {
        ".gitignore": GITIGNORE,
        "config.yaml": CONFIG.format(name=name, description=description),
        "README.md": README.format(name=name, description=description),
        f"src/{name}/main.bif": MAIN.format(name=name),
    }
    for relative, text in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return project


__all__ = ["ScaffoldError", "create"]
