"""Basic shared configuration management.

Reads shared config and makes object available for use by other modules
"""
import logging
import os
import os.path
import yaml


from ._errors import TidRuntimeError

# TODO switch to appdirs https://github.com/ActiveState/appdirs
default_config = os.path.join(
    os.path.dirname(__file__), "..", "config", "configuration.yml"
)

_GLOBAL_CONFIG = None


class Configuration:
    """Actual configuration object."""

    def __init__(self, config_file: str = default_config) -> None:
        """
        Args:
            config_file : str of the path of the config file
        """
        self.config_file = config_file

        with open(self.config_file, encoding="utf-8") as fname:
            self.conf = yaml.safe_load(fname)
            self.cache_dir = os.path.expanduser(self.conf["cache_dir"])
            self.logging = self.conf.get("logging", {})
            self.log_level = self.logging.get("level", logging.WARNING)
            logging.basicConfig(
                level=self.log_level,
                format="%(asctime)s [%(filename)s:%(lineno)d][%(levelname)s] %(message)s",
                datefmt=self.logging.get("datefmt", "%Y-%m-%d %H:%M:%S"),
            )
            self.logger = logging.getLogger("tid")

            self.credentials = self.conf.get("credentials", {})

        if self.credentials:
            if "nasa_username" in self.credentials:
                os.environ["NASA_USERNAME"] = self.credentials["nasa_username"]
            if "nasa_password" in self.credentials:
                os.environ["NASA_PASSWORD"] = self.credentials["nasa_password"]


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
