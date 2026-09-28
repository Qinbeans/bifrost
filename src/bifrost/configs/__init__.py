from pathlib import Path

import yaml

from bifrost.configs.schema import Config, source_root


class ConfigBuilder:
    def __init__(self, config_path: Path = Path("./config.yaml")) -> None:
        with config_path.open("r") as config_file:
            self._config = Config(**yaml.safe_load(config_file))
        # Library paths are relative to config.yaml, wherever bifrost runs from.
        self._config.libraries = [
            config_path.parent / library if isinstance(library, Path) and not library.is_absolute() else library
            for library in self._config.libraries
        ]

    def build(self) -> Config:
        """Build and return the configuration object."""
        return self._config


__all__ = ["Config", "ConfigBuilder", "source_root"]
