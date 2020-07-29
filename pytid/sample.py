import argparse
from datetime import datetime, timedelta
from laika import AstroDog
from laika.lib import coordinates
import logging

from pytid.utils.configuration import Configuration
from pytid.gnss import bias_solve, connections, get_data, plot

conf = Configuration()

dog = AstroDog(cache_dir=conf.gnss.get("cache_dir"))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s:%(lineno)d][%(levelname)s] %(message)s",
    datefmt=conf.logging.get("datefmt")
)

fl_stations = [
    'bkvl', 'crst', 'flbn', 'talh', 
    'fmyr', 'napl', 'okcb', 'ormd', 'pbch', 'pcla',
    'flwe', 'zefr', 'zjx1', 'tn22',
]

# near vandenberg
ca_stations = [
    "srs1", "cru1", "vndp", "p513", "ozst", "copr",
    "ana1", "cat3", "sni1", "scip", "p172", "islk",
    "p563", "p591", "azu1", "vdcy", "p231", "p300",
    "p467", "cat2", "sio5", "trak", "p588", "casn",
    "csst", "bar1", "p475", "tost", "ores", "p472",
    "p473", "p066", "p480", "p483", "p066", "p500",
    "p003"
]

"""
from geopy.geocoders import Nominatim
geocoder = Nominatim(user_agent="tid")
stations = get_data.get_nearby_stations(
    dog,
    coordinates.geodetic2ecef((*geocoder.geocode("Jiuquan, China")[1], 0)),
    dist=1.5e6
)
print("%d stations: " % len(stations))
print(" ".join(stations))
# China Chuangxin 3A (02) launch: July 4, 2020 ??:?? UTC
start_date = datetime.strptime("2019-06-12", "%Y-%m-%d")
"""

stations = ca_stations

"""
# crew dragon launch in FL: May 30, 2020, 15:22 EDT / 19:22 UTC
start_date = datetime.strptime("2020-05-30", "%Y-%m-%d")

# spacex launch in FL: February 17, 2020, 15:05 UTC
start_date = datetime.strptime("2020-02-17", "%Y-%m-%d")
"""

# spacex launch in CA: June 12, 2019, 14:17 UTC
start_date = datetime.strptime("2019-06-12", "%Y-%m-%d")


duration = timedelta(days=1)
# make a "scenario"
scenario = get_data.ScenarioInfo(dog, start_date, duration, stations)

# turn our station data into "connections" which are periods of
# signal lock without cycle slips
logger.info("Reorganizing data into connections")
conns = connections.get_connections(scenario)
# then organize it so it's easier to look up
conn_map = connections.make_conn_map(conns)


# attempt to solve integer ambiguities
logger.info("Solving ambiguities")
#connections.correct_conns(station_locs, station_data, station_clock_biases, conns)
connections.correct_conns_code(scenario, conns)

# this will get vtec data, accounting for ambiguities but NOT clock biases
logger.info("Calculating vTEC data")
station_vtecs = get_data.get_vtec_data(scenario, conn_map=conn_map)

# this attempts to find coincidences of satellites and station observations
# from which to derive biases
logger.info("Locating coincidences for biases")
cal_dat = bias_solve.gather_data(start_date, station_vtecs)

# this uses least squares to attempt to resolve said ambiguities
# XXX: you want quite a few datapoints for this to work well
logger.info("Resolving biases")
sat_biases, rcvr_biases, tecs = bias_solve.lsq_solve(dog, *cal_dat)

# now go back and update our vtecs data...
logger.info("Correcting vTEC data with biases")
corrected_vtecs = get_data.correct_vtec_data(scenario, station_vtecs, sat_biases, rcvr_biases)
