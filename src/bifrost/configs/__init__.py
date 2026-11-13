from pathlib import Path

import yaml
from pydantic import ValidationError

from bifrost.configs import features
from bifrost.configs.schema import Config, source_root


class ConfigError(ValueError):
    """A config.yaml that does not describe a project: the message names each key that is wrong or missing."""


def _problems(path: Path, error: ValidationError) -> str:
    """Each problem pydantic found, one per line, by its key's dotted path: ``flags: missing``."""
    lines = []
    for problem in error.errors():
        key = ".".join(str(part) for part in problem["loc"]) or "(top level)"
        what = "missing" if problem["type"] == "missing" else problem["msg"]
        lines.append(f"  {key}: {what}")
    return f"{path} is not a valid project configuration:\n" + "\n".join(lines)


class ConfigBuilder:
    def __init__(
        self,
        config_path: Path = Path("./config.yaml"),
        *,
        install: bool = False,
        resolve: bool = True,
        features: set[str] | None = None,
        default_features: bool = True,
    ) -> None:
        """Load ``config_path``; with ``resolve``, add the packages it uses (see ``bifrost.packages``).

        ``install``: fetch and unpack packages that are not yet (``bfc build`` does; the editor does not).
        ``features``: the project's own features to turn on, besides its defaults (unless not
        ``default_features``); see ``bifrost.configs.features``.
        """
        with config_path.open("r") as config_file:
            loaded = yaml.safe_load(config_file)
        if not isinstance(loaded, dict):
            msg = f"{config_path} is not a valid project configuration: it holds no keys (package:, flags:, ...)"
            raise ConfigError(msg)
        try:
            self._config = Config(**loaded)
        except ValidationError as error:
            raise ConfigError(_problems(config_path, error)) from None
        # Library paths are relative to config.yaml, wherever bifrost runs from.
        self._config.libraries = [
            config_path.parent / library if isinstance(library, Path) and not library.is_absolute() else library
            for library in self._config.libraries
        ]
        if not resolve:
            return
        if self._config.features or features:
            self._own_features(config_path.parent, features or set(), default_features=default_features)
        if self._config.packages:
            from bifrost import packages  # noqa: PLC0415 - packages load their own configs with this

            self._config = packages.resolve(self._config, config_path.parent, install=install)

    def _own_features(self, folder: Path, requested: set[str], *, default_features: bool) -> None:
        """Use the project's own features: its externs, libraries and files that features gate."""
        config = self._config
        features.check(config)
        config.enabled = features.enabled(config, requested, defaults=default_features, owner=config.package.name)
        config.externs, libraries, config.gated = features.apply(config, folder, config.enabled)
        config.libraries = [*config.libraries, *(library for library in libraries if library not in config.libraries)]

    def build(self) -> Config:
        """Build and return the configuration object."""
        return self._config


__all__ = ["Config", "ConfigBuilder", "ConfigError", "source_root"]
