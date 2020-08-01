"""
this file handles basic data gathering and filtering
so basically convenience functions around calculations to give
nice usable datastructures
"""
from collections import defaultdict
import copy
from datetime import datetime, timedelta
import io
import json
import math
import numpy
import os
import pickle
import requests
from scipy.signal import butter, filtfilt
import zipfile

from laika import constants
from laika.downloader import download_cors_station, download_file
from laika.gps_time import GPSTime
from laika.lib import coordinates
from laika.rinex_file import RINEXFile, DownloadError
from laika.dgps import get_station_position
import laika.raw_gnss as raw

from pytid.gnss import tec

# one day worth of samples every 30 seconds
default_len = int(24*60/0.5)

# GPS has PRN 1-32; GLONASS has 1-23 but could use more?
satellites = ['G%02d' % i for i in range(1,33)] + ['R%02d' % i for i in range(1,25)]

"""
Couldn't find good data source so used
http://geodesy.unr.edu/ who has data
citation:
Blewitt, G., W. C. Hammond, and C. Kreemer (2018),
Harnessing the GPS data explosion for interdisciplinary science, Eos, 99,
https://doi.org/10.1029/2018EO104623
"""
station_network_info = json.load(open(os.path.dirname(__file__) + "/station_networks.json"))
extra_station_info_path = os.path.dirname(__file__) + "/stations.pickle"
extra_station_info = pickle.load(open(extra_station_info_path, "rb"))

def get_satellite_delays(dog, date):
    dog.get_dcb_data(GPSTime.from_datetime(date))
    res = {}
    # published data is in nanoseconds...
    # ours is in pseudorange (meters)
    factor = constants.SPEED_OF_LIGHT/1e9
    factor = 0.365
    for prn in satellites:
        if hasattr(dog.dcbs[prn][0], 'C1W_C2W'):
            res[prn] = dog.dcbs[prn][0].C1W_C2W * factor
        elif hasattr(dog.dcbs[prn][0], 'C1P_C2P'):
            res[prn] = dog.dcbs[prn][0].C1P_C2P * factor
    return res

def get_nearby_stations(dog, point, dist=400000):
    cache_dir = dog.cache_dir
    cors_pos_path = cache_dir + 'cors_coord/cors_station_positions'
    cors_pos_dict = numpy.load(open(cors_pos_path, "rb"), allow_pickle=True).item()
    station_names = list()
    station_pos = list()

    for name, (_, pos, _) in cors_pos_dict.items():
        station_names.append(name)
        station_pos.append(pos)
    for name, pos in extra_station_info.items():
        station_names.append(name)
        station_pos.append(pos)

    station_names = numpy.array(station_names)
    station_pos = numpy.array(station_pos)
    point = numpy.array(point)

    dists = numpy.sqrt( ((station_pos - numpy.array(point))**2).sum(1) )

    return list(station_names[numpy.where(dists < dist)[0]])

def download_misc_igs_station(time, station_name, cache_dir):
    """
    Downloader for non-CORS stations
    """
    cache_subdir = cache_dir + 'misc_igs_obs/'
    t = time.as_datetime()
    # different path formats...

    folder_path = t.strftime('%Y/%j/')
    filename = station_name + t.strftime("%j0.%yo")
    url_bases = (
        'ftp://garner.ucsd.edu/archive/garner/rinex/',
        'ftp://data-out.unavco.org/pub/rinex/obs/',
    )
    try:
        filepath = download_file(url_bases, folder_path, cache_subdir, filename, compression='.Z')
        return filepath
    except IOError:
        url_bases = (
        'ftp://igs.gnsswhu.cn/pub/gps/data/daily/',
        'ftp://cddis.nasa.gov/gnss/data/daily/',
        )
        folder_path += t.strftime("%yo/")
        try:
            filepath = download_file(url_bases, folder_path, cache_subdir, filename, compression='.Z')
            return filepath
        except IOError:
            return None

def download_korean_station(time, station_name, cache_dir):
    """
    Downloader for Korean stations
    TODO: we can download from multiple stations at once and save some time here....
    """
    json_url = 'http://gnssdata.or.kr/download/createToZip.json'
    zip_url = 'http://gnssdata.or.kr/download/getZip.do?key=%d'

    cache_subdir = cache_dir + 'korean_obs/'
    t = time.as_datetime()
    # different path formats...
    folder_path = cache_subdir + t.strftime('%Y/%j/')
    filename = folder_path + station_name + t.strftime("%j0.%yo")

    if os.path.isfile(filename):
        return filename
    elif not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    start_day = t.strftime("%Y%m%d")
    postdata = {
        'corsId': station_name.upper(),
        'obsStDay':start_day,
        'obsEdDay':start_day,
        'dataTyp': 30
    }
    res = requests.post(json_url, data=postdata).text
    if not res:
        raise DownloadError
    res_dat = json.loads(res)
    if not res_dat.get('result', None):
        raise DownloadError

    key = res_dat['key']
    zipstream = requests.get(zip_url % key, stream=True)
    zipdat = zipfile.ZipFile(io.BytesIO(zipstream.content))
    for zf in zipdat.filelist:
        station = zipfile.ZipFile(io.BytesIO(zipdat.read(zf)))
        for rinex in station.filelist:
            if rinex.filename.endswith("o"):
                open(filename, "wb").write(station.read(rinex))
    return filename

def data_for_station(dog, station_name, date):
    """
    Get data from a particular station and time. Wraps a number of laika function calls.
    Station names are CORS names (eg: 'slac')
    Dates are datetimes (eg: datetime(2020,1,7))
    """
    time = GPSTime.from_datetime(date)
    rinex_obs_file = None

    # handlers for specific networks
    handlers = {
        'Korea': download_korean_station
    }

    network = station_network_info.get(station_name, None)

    # no special network, so try using whatever
    if network is None:
        try:
            station_pos = get_station_position(station_name, cache_dir=dog.cache_dir)
            rinex_obs_file = download_cors_station(time, station_name, cache_dir=dog.cache_dir)
        except (KeyError, DownloadError):
            pass

        if not rinex_obs_file:
            # station position not in CORS map, try another thing
            if station_name in extra_station_info:
                station_pos = numpy.array(extra_station_info[station_name])
                rinex_obs_file = download_misc_igs_station(time, station_name, cache_dir=dog.cache_dir)
            else:
                raise DownloadError

    else:
        station_pos = numpy.array(extra_station_info[station_name])
        rinex_obs_file = handlers[network](time, station_name, cache_dir=dog.cache_dir)

    obs_data = RINEXFile(rinex_obs_file, rate=30)
    return station_pos, raw.read_rinex_obs(obs_data)


# want to create:
# receiver -> satellite -> tick -> measurement
# from
# receiver -> tick -> measurement
def station_transform(station_data, start_dict=None, offset=0):
    '''
    Input station_data object has many interleaved observations from different satelites. Need to organize them
    into dicts grouped by satellite.
    :param station_data:
    :param start_dict:
    :param offset:
    :return:
    '''
    # TODO GPS only
    if start_dict is None:
        ret = {prn: defaultdict() for prn in satellites}
    else:
        ret = start_dict

    for i, dat in enumerate(station_data):
        for sample in dat:
            # ignore non-GPS/GLONASS data for now
            if sample.prn[0] in {"G", "R"}:
                ret[sample.prn][i + offset] = sample

    return ret

def empty_factory():
    return None

class ScenarioInfo:
    def __init__(self, dog, start_date, duration, stations):
        self.dog = dog
        self.start_date = start_date
        self.duration = duration
        self.stations = stations
        self._station_locs = None
        self._station_data = None
        self._clock_biases = None

        # Local coordinates for the stations we're using
        # this enables faster elevation lookups
        self.station_converters = dict()

    @property
    def station_locs(self):
        if self._station_locs is None:
            self.populate_data()
        return self._station_locs

    @property
    def station_data(self):
        if self._station_data is None:
            self.populate_data()
        return self._station_data

    def station_el(self, station, sat_pos):
        '''
        Re-use station converters for faster elevation lookups:
        we do this operation a lot, and Laika creates objects each time
        which is quite expensive
        '''
        if station not in self.station_converters:
            self.station_converters[station] = (
                coordinates.LocalCoord.from_ecef(self.station_locs[station])
            )
        sat_ned = self.station_converters[station].ecef2ned(sat_pos)
        sat_range = numpy.linalg.norm(sat_ned)
        return numpy.arcsin(-sat_ned[2]/sat_range)

    def populate_data(self):
        self._station_locs = {}
        self._station_data = {}
        bad_stations = []
        for station in self.stations:
            print(station)
            cache_name = "cached/stationdat_%s_%s_to_%s" % (
                station,
                self.start_date.strftime("%Y-%m-%d"),
                (self.start_date + self.duration).strftime("%Y-%m-%d")
            )
            if os.path.exists(cache_name):
                self.station_data[station] = pickle.load(open(cache_name, "rb"))
                try:
                    self.station_locs[station] = get_station_position(station, cache_dir=self.dog.cache_dir)
                except KeyError:
                    self.station_locs[station] = numpy.array(extra_station_info[station])
                continue

            self.station_data[station] = {prn: defaultdict(empty_factory) for prn in satellites}
            date = self.start_date
            while date < self.start_date + self.duration:
                try:
                    loc, data = data_for_station(self.dog, station, date)
                    self.station_data[station] = station_transform(
                                                data,
                                                start_dict=self.station_data[station],
                                                offset=int((date - self.start_date).total_seconds()/30)
                                            )
                    self.station_locs[station] = loc
                except (ValueError, DownloadError):
                    print("*** error with station " + station)
                    bad_stations.append(station)
                date += timedelta(days=1)
            os.makedirs("cached", exist_ok=True)
            pickle.dump(self.station_data[station], open(cache_name, "wb"))
        for bad_station in bad_stations:
            self.stations.remove(bad_station)

    @property
    def clock_biases(self):
        if self._clock_biases is None:
            self.populate_station_clock_biases()
        return self._clock_biases

    def populate_station_clock_biases(self):
        """
        Figure out the clock bias for the station at each tick.
        For each tick, take the satellite position - station position, and after
        correcting for the satellite bias, assume the remainder is receiver bias.
        Average it out a bit
        """
        self._clock_biases = dict()
        for station in self.station_data:
            self._clock_biases[station] = dict()
            max_tick = max(max(self.station_data[station][prn].keys(), default=0) for prn in satellites)
            for tick in range(max_tick):
                diffs = []
                for prn in satellites:
                    if self.station_data[station][prn][tick] is None:
                        continue
                    if math.isnan(self.station_data[station][prn][tick].observables['C1C']):
                        continue
                    if not self.station_data[station][prn][tick].corrected:
                        self.station_data[station][prn][tick].correct(
                            self.station_locs[station],
                            self.dog
                        )
                    if math.isnan(self.station_data[station][prn][tick].sat_pos_final[0]):
                        continue
                    diffs.append(
                        (
                            numpy.linalg.norm(
                                self.station_data[station][prn][tick].sat_pos_final
                                - self.station_locs[station]
                            ) - self.station_data[station][prn][tick].observables['C1C']
                        ) / tec.C - self.station_data[station][prn][tick].sat_clock_err
                    )
                    assert not math.isnan(diffs[-1])
                if diffs:
                    self._clock_biases[station][tick] = numpy.median(diffs)


def get_vtec_data(scenario, conn_map=None, biases=None):
    '''
    Iterates over (station, PRN) pairs and computes the VTEC for each one. VTEC here takes the form of a tuple of
    lists in the form: (locs, dats, slants) where each one is a list of values of length max-tick-for-station-prn.
    In other words, for every tick in the range of ticks seen for the (station, PRN) pair, we have a (loc, dat, slant)
    triple, although they are each in their own vector.
    Here 'loc' is the location of the ionosphere starting point. 'data' is the vtec calculation, and 'slatn' is the
    slant_to_vertical conversion factor.
    :param scenario:
    :param conn_map:
    :param biases:
    :return:
    '''
    station_vtecs = defaultdict(dict)   # The eventual output
    def vtec_for(station, prn, conns=None, biases=None):
        if biases:
            station_bias = biases.get(station, 0)
            sat_bias = biases.get(prn, 0)
        else:
            station_bias, sat_bias = 0, 0
        # Only bother if the particular (station, prn) has not been done yet:
        if prn not in station_vtecs[station]:
            dats = []
            locs = []
            slants = []
            if scenario.station_data[station].get(prn):
                end = max(scenario.station_data[station][prn].keys())
            else:
                end = 0
            for i in range(end):    # iterate over integers in the range of ticks
                measurement = scenario.station_data[station][prn][i]
                # if conns specified, require ambiguity data
                if conns:
                    if measurement and conns[i] and (conns[i].n1 or conns[i].offset): # and numpy.std(conns[i].n1s) < 3:
                        res = tec.calc_vtec(
                            scenario,
                            station, prn, i,
                            n1=conns[i].n1,
                            n2=conns[i].n2,
                            rcvr_bias=station_bias,
                            sat_bias=sat_bias,
                            offset=conns[i].offset,
                        )
                        if res is None:
                            locs.append(None)
                            dats.append(math.nan)
                            slants.append(math.nan)
                        else:
                            dats.append(res[0])
                            locs.append(res[1])
                            slants.append(res[2])
                    else:
                        locs.append(None)
                        dats.append(math.nan)
                        slants.append(math.nan)

                elif measurement:
                    res = tec.calc_vtec(scenario, station, prn, i)
                    if res is None:
                        locs.append(None)
                        dats.append(math.nan)
                        slants.append(math.nan)
                    else:
                        dats.append(res[0])
                        locs.append(res[1])
                        slants.append(res[2])
                else:
                    locs.append(None)
                    dats.append(math.nan)
                    slants.append(math.nan)

            station_vtecs[station][prn] = (locs, dats, slants)
        return station_vtecs[station][prn]

    for station in scenario.stations:
        print(station)
        for prn in satellites:
            # Use the connection map if we have it, otherwise don't.
            if conn_map:
                if station not in conn_map:
                    break  # no connections... ignore this
                vtec_for(station, prn, conns=conn_map[station][prn], biases=biases)
            else:
                vtec_for(station, prn, biases=biases)
    return station_vtecs

def correct_vtec_data(scenario, vtecs, sat_biases, station_biases):
    corrected = copy.deepcopy(vtecs)
    bad_stations = []
    for station in corrected:
        if station not in station_biases:
            bad_stations.append(station)
            continue
        for prn in satellites:
            if prn not in corrected[station]:
                print("no sat info for %s for %s" % (station, prn))
                continue
            for i in range(len(corrected[station][prn][0])):
                dat = corrected[station][prn][0][i], corrected[station][prn][1][i], corrected[station][prn][2][i]
                if dat[0] is None:
                    continue
                if prn[0] == 'G':
                    rcvr_bias = station_biases[station][0]
                elif prn[0] == 'R':
                    chan = scenario.dog.get_glonass_channel(prn, scenario.station_data[station][prn][i].recv_time)
                    rcvr_bias = station_biases[station][1] + station_biases[station][2] * chan
                dat = tec.correct_tec(dat, rcvr_bias=rcvr_bias, sat_bias=sat_biases[prn])
                corrected[station][prn][0][i], corrected[station][prn][1][i], corrected[station][prn][2][i] = dat
    for station in bad_stations:
        print("missing bias data for %s: deleting vtecs" % station)
        del corrected[station]
    return corrected


# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    a, b = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
    if len(data) < 28:
        return [math.nan] * len(data)
    return filtfilt(a, b, data)


def bpfilter(data, short_min=2, long_min=12):
    return butter_bandpass_filter(data, 1/(long_min*60), 1/(short_min*60), 1/30)

def reassemble_chunks(chunked, expected_len=default_len):
    res = numpy.full(expected_len, math.nan)
    for idx, chunk in chunked:
        res[idx:idx+len(chunk)] = numpy.array(chunk)
    return res

def get_contiguous(data, min_length=28):
    i = 0
    runs = []
    while i < len(data):
        if data[i] and not math.isnan(data[i]):
            run = [data[i]]
            for j in range(i+1, len(data)):
                if not (data[j] and not math.isnan(data[j])):
                    break
                run.append(data[j])
            if len(run) > min_length:
                runs.append( (i, run) )
            i += len(run)
        i += 1
    return runs

def filter_contiguous(data, short_min=2, long_min=12):
    expected_len = len(data)
    chunked = get_contiguous(data)
    filtered_chunks = []
    for idx, chunk in chunked:
        filtered_dat = bpfilter(chunk, short_min=short_min, long_min=long_min)
        filtered_chunks.append( (idx, filtered_dat) )
    return reassemble_chunks(filtered_chunks, expected_len=expected_len)

def remove_slips(data_stream, slip_value=3):
    for i in range(1, len(data_stream)):
        if abs(data_stream[i] - data_stream[i-1]) > slip_value:
            data_stream[i] = math.nan

def barrel_roll(data_stream, radius=1, tau0=120*2, zeta0=40):
    """
    Can't believe "do a barrel roll" is science, but
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JA021723
    """
    contact_point = (0, data_stream[0])

    def calc_delta(contact_point, next_i):
        dx = (next_i - contact_point[0])/tau0
        dy = (data_stream[next_i] - contact_point[1])/zeta0
        reach = math.sqrt(dx**2 + dy**2)/(2*radius)
        if reach > 1:
            # datapoint is outside the circle's reach
            # beta = 90 degrees
            beta = math.pi/2
        else:
            beta = math.asin(reach)
        theta = math.atan(dy/dx)
        return beta - theta

    def next_contact_point(contact_point):
        min_idx = None
        min_delta = math.inf
        for i in range(
            contact_point[0] + 1,
            min(
                contact_point[0] + math.ceil(2*radius*tau0),
                len(data_stream) - 1
            )
        ):
             delta = calc_delta(contact_point, i)
             if delta < min_delta:
                 min_delta = delta
                 min_idx = i
        if min_idx is None:
            return None
        return (min_idx, data_stream[min_idx])

    points = [contact_point]
    while points[-1][0] < len(data_stream):
        next_point = next_contact_point(points[-1])
        if next_point:
            points.append(next_point)
        else:
            break
    return points

def get_brc(data_stream, radius=1, tau0=120*2, zeta0=40):
    brc_points = barrel_roll(data_stream, radius=radius, tau0=tau0, zeta0=zeta0)
    xs = [x[0] for x in brc_points]
    ys = [x[1] for x in brc_points]
    return numpy.interp(range(len(data_stream)), xs, ys)

def get_depletion(signal):
    # TODO use the bands of -3TEC to +1TEC to keep waves?
    return get_brc(signal) - signal


def depletion_contiguous(data):
    expected_len = len(data)
    chunked = get_contiguous(data)
    filtered_chunks = []
    for idx, chunk in chunked:
        filtered_dat = get_depletion(chunk)
        filtered_chunks.append( (idx, filtered_dat) )
    return reassemble_chunks(filtered_chunks, expected_len=expected_len)