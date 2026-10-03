from pathlib import Path

import yaml

from bifrost.configs.schema import Config, source_root


class ConfigBuilder:
    def __init__(
        self, config_path: Path = Path("./config.yaml"), *, install: bool = False, resolve: bool = True
    ) -> None:
        """Load ``config_path``; with ``resolve``, add the packages it uses (see ``bifrost.packages``).

        ``install``: fetch and unpack packages that are not yet (``bfc build`` does; the editor does not).
        """
        with config_path.open("r") as config_file:
            self._config = Config(**yaml.safe_load(config_file))
        # Library paths are relative to config.yaml, wherever bifrost runs from.
        self._config.libraries = [
            config_path.parent / library if isinstance(library, Path) and not library.is_absolute() else library
            for library in self._config.libraries
        ]
        if resolve and self._config.packages:
            from bifrost import packages  # noqa: PLC0415 - packages load their own configs with this

            self._config = packages.resolve(self._config, config_path.parent, install=install)

    def build(self) -> Config:
        """Build and return the configuration object."""
        return self._config


__all__ = ["Config", "ConfigBuilder", "source_root"]
