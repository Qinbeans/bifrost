import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich import print  # noqa: A004
from rich.markup import escape
from typer import Typer

from bifrost import packages, scaffold, versioning
from bifrost.configs import Config, ConfigBuilder, source_root
from bifrost.formatter import FormatError, format_source
from bifrost.lowering import BifrostError, lower_file
from bifrost.packages import PackageError
from bifrost.project import Project

cli = Typer()


@cli.callback()
def main() -> None:
    """Compile Bifrost programs."""


@cli.command()
def init(
    name: Annotated[str, typer.Argument(help="Project name, in snake_case; also its folder.")],
    description: Annotated[str, typer.Option("--description", "-d", help="One line about the project.")] = "",
) -> None:
    """Create a project: config.yaml, .gitignore, README.md, and src/<name>/main.bif."""
    try:
        project = scaffold.create(name, Path.cwd(), description)
    except scaffold.ScaffoldError as error:
        print(f"[bold red]✘ ERROR[/bold red]: {escape(str(error))}", file=sys.stderr)
        raise typer.Exit(1) from None
    print(f"[bold green]✔ SUCCESS[/bold green]: created {project}")
    print(f"  cd {name}\n  bfc build\n  ./build/{name}")


@cli.command()
def build(
    source: Annotated[Path | None, typer.Argument(help="File to compile; default: the entry in config.yaml.")] = None,
    config: Annotated[Path, typer.Option("--config", "-c", help="Project configuration.")] = Path("config.yaml"),
    path: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output folder; default: build/ next to config.yaml.")
    ] = None,
) -> None:
    """Compile a Bifrost source file (by default, the project's entry) into the configured executable."""
    config_path = config
    try:
        config = ConfigBuilder(config_path, install=True).build()  # fetching and unpacking its packages
    except FileNotFoundError:
        msg = f"no {config_path}; create a project with `bfc init <name>`"
        print(f"[bold red]✘ ERROR[/bold red]: {escape(msg)}", file=sys.stderr)
        raise typer.Exit(1) from None
    except PackageError as error:
        print(f"[bold red]✘ ERROR[/bold red]: {escape(str(error))}", file=sys.stderr)
        raise typer.Exit(1) from None
    project_folder = config_path.parent
    if config.package.build == "library":
        msg = (
            f"{config_path} is a library (package.build: library), which builds no executable; "
            "`bfc package` packages it, and projects use it from their packages:"
        )
        print(f"[bold red]✘ ERROR[/bold red]: {escape(msg)}", file=sys.stderr)
        raise typer.Exit(1)
    if source is None:
        if config.package.entry is None:
            msg = f"give a file to build, or set package.entry in {config_path}"
            print(f"[bold red]✘ ERROR[/bold red]: {escape(msg)}", file=sys.stderr)
            raise typer.Exit(1)
        source = project_folder / config.package.entry
    config.path = path or project_folder / "build"
    config.path.mkdir(parents=True, exist_ok=True)
    project = Project(config)
    try:
        unit = lower_file(project, source, root=source_root(project_folder))
        with unit.errors():
            executable = project.build()
    except BifrostError as error:
        print(f"[bold red]✘ ERROR[/bold red]: {escape(str(error))}", file=sys.stderr)
        raise typer.Exit(1) from None
    except (FileNotFoundError, RuntimeError) as error:  # a missing library, or a failed link
        print(f"[bold red]✘ ERROR[/bold red]: {escape(str(error))}", file=sys.stderr)
        raise typer.Exit(1) from None
    print(f"[bold green]✔ SUCCESS[/bold green]: built {executable}")


@cli.command()
def package(
    config: Annotated[Path, typer.Option("--config", "-c", help="Project configuration.")] = Path("config.yaml"),
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Folder for the .bifpkg; default: dist/.")
    ] = None,
) -> None:
    """Package the project for others to use: its sources, config.yaml and built C libraries, as a .bifpkg."""
    if not config.is_file():
        print(f"[bold red]✘ ERROR[/bold red]: no {escape(str(config))}; package a project folder", file=sys.stderr)
        raise typer.Exit(1)
    try:
        archive = packages.build(config.parent, output)
    except PackageError as error:
        print(f"[bold red]✘ ERROR[/bold red]: {escape(str(error))}", file=sys.stderr)
        raise typer.Exit(1) from None
    print(f"[bold green]✔ SUCCESS[/bold green]: packaged {archive}")


@cli.command()
def fmt(
    sources: Annotated[list[Path], typer.Argument(help="Files to format in place.")],
    *,
    check: Annotated[bool, typer.Option("--check", help="Only report files that are not formatted.")] = False,
) -> None:
    """Format Bifrost source files."""
    unformatted = 0
    for source in sources:
        text = source.read_text()
        try:
            formatted = format_source(text)
        except FormatError as error:
            typer.echo(f"{source}: {error}", err=True)
            raise typer.Exit(1) from None
        if formatted == text:
            continue
        unformatted += 1
        if check:
            typer.echo(f"would reformat {source}")
        else:
            source.write_text(formatted)
            typer.echo(f"reformatted {source}")
    if check and unformatted:
        raise typer.Exit(1)


@cli.command()
def version(
    value: Annotated[str | None, typer.Argument(help="Set the version to this, like 1.2.0.")] = None,
    bump: Annotated[
        versioning.Bump | None, typer.Option("--bump", help="Bump this part (the parts after it reset to 0).")
    ] = None,
    config: Annotated[Path, typer.Option("--config", "-c", help="Project configuration.")] = Path("config.yaml"),
    *,
    short: Annotated[bool, typer.Option("--short", help="Print only the version.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the new version without writing it.")] = False,
) -> None:
    """Print the project's version, or set or bump it (package.version in config.yaml)."""
    if value is not None and bump is not None:
        print("[bold red]✘ ERROR[/bold red]: give a version or --bump, not both", file=sys.stderr)
        raise typer.Exit(1)
    try:
        name, current = versioning.read(config)
        new = value if value is not None else versioning.bumped(current, bump) if bump is not None else None
        if new is not None and not dry_run:
            versioning.write(config, new)
    except FileNotFoundError:
        print(f"[bold red]✘ ERROR[/bold red]: no {config}; create a project with `bfc init <name>`", file=sys.stderr)
        raise typer.Exit(1) from None
    except versioning.VersionError as error:
        print(f"[bold red]✘ ERROR[/bold red]: {escape(str(error))}", file=sys.stderr)
        raise typer.Exit(1) from None
    if new is None:
        typer.echo(current if short else f"{name} {current}")
    else:
        typer.echo(new if short else f"{name} {current} => {new}")


@cli.command()
def lsp() -> None:
    """Run the Bifrost language server over stdio."""
    from bifrost.server import serve  # noqa: PLC0415 - only the server needs pygls

    serve()


config_cli = Typer(help="Manage a project's config.yaml.")
cli.add_typer(config_cli, name="config")


@config_cli.command()
def traverse(  # noqa: PLR0913 - each argument is a command-line option
    inputs: Annotated[str, typer.Option("--input", "-i", help="Comma-separated C/C++ headers or directories.")],
    config: Annotated[Path, typer.Option("--config", "-c", help="Project configuration to update.")] = Path(
        "config.yaml"
    ),
    include_dirs: Annotated[
        list[Path] | None, typer.Option("--include-dir", "-I", help="Extra header search path (repeatable).")
    ] = None,
    clang_args: Annotated[
        list[str] | None, typer.Option("--clang-arg", help="Extra argument for clang, e.g. -DPLATFORM_DESKTOP.")
    ] = None,
    *,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report what would change without writing.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="List every skipped declaration.")] = False,
) -> None:
    """Generate extern declarations in config.yaml from C/C++ headers, one module per header."""
    from bifrost.configs.traverse import TraverseError, dump, merge  # noqa: PLC0415 - only this command needs libclang
    from bifrost.configs.traverse import traverse as traverse_headers  # noqa: PLC0415

    paths = [Path(part.strip()) for part in inputs.split(",") if part.strip()]
    try:
        modules = traverse_headers(paths, include_dirs or [], clang_args or [])
    except TraverseError as error:
        print(f"[bold red]✘ ERROR[/bold red]: {escape(str(error))}", file=sys.stderr)
        raise typer.Exit(1) from None

    data = yaml.safe_load(config.read_text()) or {}
    updated = merge(data, modules)
    try:
        Project(Config(**updated))  # declares every extern, as a build would
    except Exception as error:  # noqa: BLE001 - any failure means the result must not be written
        print(f"[bold red]✘ ERROR[/bold red]: the generated externs do not load: {escape(str(error))}", file=sys.stderr)
        raise typer.Exit(1) from None

    for module in modules:
        if module.error is not None:
            print(f"[bold yellow]! {module.module}[/bold yellow] ({module.header.name}): not parsed, left unchanged")
            print(f"[dim]{escape(str(module.error))}[/dim]")
            continue
        after = f", parsed after {', '.join(h.name for h in module.context)}" if module.context else ""
        print(
            f"[bold]{module.module}[/bold] ({module.header.name}{after}): {len(module.functions)} functions, "
            f"{len(module.structs)} structs, {len(module.skipped)} skipped"
        )
        for skipped in module.skipped if verbose else []:
            print(f"  [dim]skipped {skipped.name}: {escape(skipped.reason)}[/dim]")
    if not verbose and any(module.skipped for module in modules):
        print("[dim]Use --verbose to see why each declaration was skipped.[/dim]")
    if dry_run:
        print(f"[yellow]dry run[/yellow]: {config} not written")
        return
    config.write_text(dump(updated))
    print(f"[bold green]✔ SUCCESS[/bold green]: updated {config}")


if __name__ == "__main__":
    cli()
