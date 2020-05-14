from datetime import datetime, timedelta
from laika import AstroDog
import os

from gnss import bias_solve, connections, get_data, plot, tec


dog = AstroDog(cache_dir=os.environ['HOME'] + "/.gnss_cache/")
stations = ['flwe', 'tn22', 'zjx1']
start_date = datetime(2020, 2, 17)
duration = timedelta(days=1)

# ask Laika for station location data + GNSS data from our stations
print("Gathering GNSS data from stations...")
station_locs, station_data = get_data.populate_data(dog, start_date, duration, stations)

# turn our station data into "connections" which are periods of
# signal lock without cycle slips
print("Reorganizing data into connections")
conns = connections.get_connections(dog, station_locs, station_data)
# then organize it so it's easier to look up
conn_map = connections.make_conn_map(conns)

# attempt to solve integer ambiguities
print("Solving ambiguities")
connections.correct_conns(station_locs, station_data, conns)

# this will get vtec data, accounting for ambiguities but NOT clock biases
print("Calculating vTEC data")
station_vtecs = get_data.get_vtec_data(dog, station_locs, station_data, conn_map=conn_map)

# this attempts to find coincidences of satellites and station observations
# from which to derive biases
print("Locating coincidences for biases")
cal_dat = bias_solve.gather_data(station_vtecs)

# this uses least squares to attempt to resolve said ambiguities
# XXX: you want quite a few datapoints for this to work well
print("Resolving biases")
sat_biases, rcvr_biases, tecs = bias_solve.lsq_solve(*cal_dat)

# now go back and update our vtecs data...
print("Correcting vTEC data with biases")
corrected_vtecs = get_data.correct_vtec_data(station_vtecs, sat_biases, rcvr_biases)

print("Showing some data")
plot.plot_station(corrected_vtecs, 'flwe')