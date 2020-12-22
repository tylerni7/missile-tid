import os, logging
import yaml
# import pytid.utils.io as myio

# start with the abosolute path in case we run it from other places...
missile_tid_rootfold = os.path.abspath(os.path.join(os.path.split(os.path.abspath(__file__))[0], '..', '..'))
default_config = os.path.join(missile_tid_rootfold, "config", "configuration.yml")




class Configuration:

    def __init__(self, config_file=default_config):
        self.config_file = config_file
        self.missile_tid_root = missile_tid_rootfold
        self.reload()

    def reload(self):
        with open(self.config_file, "r") as f:
            conf = yaml.safe_load(f)

        self._conf = conf
        self.gnss = conf.get("gnss")
        self.logging = conf.get("logging")
        self.plotting = conf.get("plotting")
        self.load_all_stations()

    def load_all_stations(self):
        with open(self.gnss.get("full_station_list"), 'r') as fsl:
            self.all_stations = list(map(lambda x: x.strip(), fsl.readlines()))


