"""The packages in ``extras/``, for CI: which to build, and the index of those released.

Each folder in ``extras/`` whose ``config.yaml`` says ``package.build: library``
is a package. It is released as ``<name>-v<version>``: a
GitHub release holding its ``.bifpkg`` files, one per target (or one ``any``,
when it links no C libraries of its own).

Usage::

    python tools/package_index.py plan [--all]     # the build matrix, as JSON
    gh api --paginate --slurp repos/OWNER/REPO/releases | python tools/package_index.py index > index.html

``plan`` lists the packages whose version has no release tag yet (``--all``:
every package, to check that they build). ``index`` writes one HTML page of
links to every ``.bifpkg`` on a published release, each with its SHA-256: what
a project's ``index:`` names, and what ``bfc build`` reads.
"""

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

EXTRAS = Path(__file__).resolve().parent.parent / "extras"
ARCHIVE = ".bifpkg"
# The Linux targets packages are built for, and the runners that build them.
TARGETS = [
    {"arch": "x86_64", "runner": "ubuntu-24.04"},
    {"arch": "aarch64", "runner": "ubuntu-24.04-arm"},
]


def packages() -> list[dict[str, Any]]:
    """Every package in ``extras/``: its name, version, folder, and whether it links C libraries it builds."""
    found = []
    for config in sorted(EXTRAS.glob("*/config.yaml")):
        raw = yaml.safe_load(config.read_text()) or {}
        package = raw.get("package") or {}
        if package.get("build") != "library":
            continue  # an executable: nothing to release
        native = any("/" in str(library) for library in raw.get("libraries", []))
        found.append(
            {
                "name": package["name"],
                "version": str(package["version"]),
                "folder": config.parent.relative_to(EXTRAS.parent).as_posix(),
                "native": native,
            }
        )
    return found


def released(tag: str) -> bool:
    """Whether ``tag`` exists on the remote: the package's version is released already."""
    result = subprocess.run(  # noqa: S603 - git, with a tag made from a config.yaml in this repository
        ["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}"],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 2}:  # 2: no such tag
        raise RuntimeError(result.stderr.decode())
    return result.returncode == 0


def plan(*, every: bool) -> dict[str, Any]:
    """Return the build matrix (a job per package and target) and the releases it makes."""
    builds, releases = [], []
    for package in packages():
        tag = f"{package['name']}-v{package['version']}"
        if not every and released(tag):
            continue
        releases.append({"name": package["name"], "version": package["version"], "tag": tag})
        targets = TARGETS if package["native"] else TARGETS[:1]  # `any`: one archive serves every target
        builds += [{**package, **target, "tag": tag} for target in targets]
    return {"builds": builds, "releases": releases}


def archive_links(releases: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(filename, url) for every ``.bifpkg`` on a published release, newest first."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for release in releases:
        if release["draft"]:
            continue
        for asset in release["assets"]:
            name: str = asset["name"]
            if not name.endswith(ARCHIVE) or name in seen:
                continue
            seen.add(name)
            url: str = asset["browser_download_url"]
            digest: str | None = asset.get("digest")
            if digest and digest.startswith("sha256:"):
                url += "#sha256=" + digest.removeprefix("sha256:")
            links.append((name, url))
    return links


def render(links: list[tuple[str, str]]) -> str:
    """Return the index page, with the archives grouped by package."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for name, url in links:
        grouped.setdefault(name.split("-")[0], []).append((name, url))
    sections = "\n".join(
        f"<h2>{html.escape(package)}</h2>\n"
        + "\n".join(f'<a href="{html.escape(url)}">{html.escape(name)}</a><br>' for name, url in archives)
        for package, archives in sorted(grouped.items())
    )
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Bifrost packages</title></head>
<body>
<h1>Bifrost packages</h1>
<p>Use them from a project's <code>config.yaml</code>: <code>index: {{this page's URL}}</code>,
then <code>packages: {{name: version}}</code>.</p>
{sections}
</body>
</html>
"""


def read_releases(text: str) -> list[dict[str, Any]]:
    """Return every release in ``text``.

    It holds one or more JSON documents, each a list of releases or, from
    ``gh api --paginate --slurp``, a list of pages.
    """
    decoder = json.JSONDecoder()
    releases: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        data, position = decoder.raw_decode(text, position)
        for item in data:
            releases.extend(item if isinstance(item, list) else [item])
    return releases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="the build matrix, as JSON")
    planning.add_argument("--all", action="store_true", help="every package, released or not")
    commands.add_parser("index", help="the index page of the releases on stdin")
    arguments = parser.parse_args()
    if arguments.command == "plan":
        sys.stdout.write(json.dumps(plan(every=arguments.all)) + "\n")
    else:
        sys.stdout.write(render(archive_links(read_releases(sys.stdin.read()))))


if __name__ == "__main__":
    main()
