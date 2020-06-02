import os, logging
import yaml

default_config = os.path.join("config", "configuration.yml")




class Configuration:

    def __init__(self, config_file=default_config):
        self.config_file = config_file
        self.reload()

    def reload(self):
        with open(self.config_file, "r") as f:
            conf = yaml.safe_load(f)

        self._conf = conf
        self.gnss = conf.get("gnss")
        self.logging = conf.get("logging")
        self.plotting = conf.get("plotting")
