from laika import AstroDog

from tid import get_data, util, scenario
from tid.config import Configuration

# load configuration data
conf = Configuration()

# create our helpful astro dog
dog = AstroDog(cache_dir=conf.cache_dir)

# time of interest for our thing
date = util.datetime_fromstr("2019-06-12")

sc = scenario.Scenario(date, 1 * util.DAYS, ["slac", "flwe"])