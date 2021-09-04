"""Basic shared configuration management.

Reads shared config and makes object available for use by other modules
"""
import os
import yaml
from pathlib import Path
from typeing import Union

from ._errors import TidRuntimeError

# TODO switch to appdirs https://github.com/ActiveState/appdirs
default_config = os.path.join(
    os.path.dirname(__file__), "..", "config", "configuration.yml"
)

_GLOBAL_CONFIG = None


class Configuration:
    """Actual configuration object."""

    def __init__(self, config_file: Union[str, os.PathLike] = default_config) -> None:
        """
        Parameters
        ----------
        config_file : str or
        """
        self.config_file = config_file

        with open(self.config_file, encoding="utf-8") as fname:
            self.conf = yaml.safe_load(fname)
            self.cache_dir = Path(self.conf["cache_dir"]).expanduser()


def set_global_config(config: Configuration) -> None:
    """
    Sets the global configuration object.

    This is for convenience to not have to pass a config option into
    ever object / call.

    """
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config


def get_global_config() -> Configuration:
    """
    Gets the global configuration object.

    This is for convenience to not have to pass a config option into
    ever object / call.

    """
    if _GLOBAL_CONFIG is None:
        raise TidRuntimeError("You have not set a global configuration")

    return _GLOBAL_CONFIG
