"""
Basic shared configuration management.
Reads shared config and makes object available for use by other modules
"""
import os
import yaml

default_config = os.path.join(
    os.path.dirname(__file__), "..", "config", "configuration.yml"
)


class Configuration:
    """
    Actual configuration object
    """

    def __init__(self, config_file: str = default_config) -> None:
        self.config_file = config_file

        with open(self.config_file, encoding="utf-8") as fname:
            self.conf = yaml.safe_load(fname)
            self.cache_dir = os.path.expanduser(self.conf.get("cache_dir"))
