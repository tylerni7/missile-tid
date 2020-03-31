# get observables at particular location (CORS station) at a particular time

from laika import AstroDog
from datetime import datetime
from laika.gps_time import GPSTime
from laika.downloader import download_cors_station
from laika.rinex_file import RINEXFile
from laika.dgps import get_station_position
import laika.raw_gnss as raw

def data_for_station(dog, station_name, date=None):
    """
    Get data from a particular station and time.
    Station names are CORS names (eg: 'slac')
    Dates are datetimes (eg: datetime(2020,1,7))
    """

    if date is None:
        date = datetime(2020,1,7)
    time = GPSTime.from_datetime(date)
    rinex_obs_file = download_cors_station(time, station_name, dog.cache_dir)

    obs_data = RINEXFile(rinex_obs_file)
    station_pos = get_station_position(station_name)
    return station_pos, raw.read_rinex_obs(obs_data)