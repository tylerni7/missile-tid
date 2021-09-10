"""
Plot data from a Falcon 9 launch into SSO from Vandenberg AFB
"""

from laika import AstroDog

from tid import plot, util, scenario
from tid.config import Configuration

# load configuration data
conf = Configuration()

# create our helpful astro dog
dog = AstroDog(cache_dir=conf.cache_dir)

# time of interest for our thing
date = util.datetime_fromstr("2019-06-12")

# near vandenberg
# fmt: off
ca_stations = [
    "srs1", "cru1", "vndp", "p513", "ozst", "copr",
    "ana1", "cat3", "sni1", "scip", "p172", "islk",
    "p563", "p591", "azu1", "vdcy", "p231", "p300",
    "p467", "sio5", "trak", "p588", "casn", "tono",
    "csst", "bar1", "p475", "tost", "ores", "p472",
    "p473", "p066", "p480", "p483", "p066", "p500",
    "p003", "gol2", "p600", "p463", "p572", "p091",
    "nvag", "p651", "p523", "jplm", "sbcc", "bill",
    "guax", "farb", "p277", "slac", "mhcb", "p217",
]
# fmt: on

conf.logger.debug("Starting scenario (downloading files, etc)")
sc = scenario.Scenario.from_daterange(date, 1 * util.DAYS, ca_stations, dog)

conf.logger.debug("Downloading complete, creating connections")
sc.make_connections()

conf.logger.debug("Connections created, resolving biases")
sc.solve_biases()

conf.logger.debug("Preparing animation")
extent = [-128, -115, 29.6, 38.1]
plot.plot_map(sc, extent=extent, frames=range(1600, 1800))
