"""
this file handles basic data gathering and filtering
so basically convenience functions around calculations to give
nice usable datastructures
"""
from collections import defaultdict
import copy
from datetime import datetime, timedelta
from scipy.signal import butter, lfilter, filtfilt, sosfiltfilt
import math
import numpy
import os
import pickle

from laika import AstroDog, constants
from laika.gps_time import GPSTime
from laika.downloader import download_cors_station
from laika.rinex_file import RINEXFile, DownloadError
from laika.dgps import get_station_position
import laika.raw_gnss as raw

from . import tec

# one day worth of samples every 30 seconds
default_len = int(24*60/0.5)

def get_satellite_delays(dog, date):
    dog.get_dcb_data(GPSTime.from_datetime(start_date))
    res = {}
    # published data is in nanoseconds...
    # ours is in pseudorange (meters)
    factor = constants.SPEED_OF_LIGHT/1e9
    factor = 0.365
    for prn in ['G%02d' % i for i in range(1, 33)]:
        if hasattr(dog.dcbs[prn][0], 'C1W_C2W'):
            res[prn] = dog.dcbs[prn][0].C1W_C2W * factor
        elif hasattr(dog.dcbs[prn][0], 'C1P_C2P'):
            res[prn] = dog.dcbs[prn][0].C1P_C2P * factor
    return res

def data_for_station(dog, station_name, date=None):
    """
    Get data from a particular station and time.
    Station names are CORS names (eg: 'slac')
    Dates are datetimes (eg: datetime(2020,1,7))
    """

    if date is None:
        date = datetime(2020,1,7)
    time = GPSTime.from_datetime(date)
    rinex_obs_file = download_cors_station(time, station_name, cache_dir=dog.cache_dir)

    obs_data = RINEXFile(rinex_obs_file)
    station_pos = get_station_position(station_name, cache_dir=dog.cache_dir)
    return station_pos, raw.read_rinex_obs(obs_data)

# want to create:
# receiver -> satellite -> tick -> measurement
# from
# receiver -> tick -> measurement
def station_transform(station_data, start_dict=None, offset=0):
    # TODO GPS only
    if start_dict is None:
        ret = {'G%02d' % i: defaultdict() for i in range(1, 33)}
    else:
        ret = start_dict

    for i, dat in enumerate(station_data):
        for sample in dat:
            # ignore non-GPS data for now
            if sample.prn.startswith("G"):
                ret[sample.prn][i + offset] = sample
    
    return ret

def empty_factory():
    return None

def populate_data(dog, start_date, duration, stations):
    station_locs = {}
    station_data = {}
    for station in stations:
        print(station)
        cache_name = "cached/stationdat_%s_%s_to_%s" % (
            station,
            start_date.strftime("%Y-%m-%d"),
            (start_date + duration).strftime("%Y-%m-%d")
        )
        if os.path.exists(cache_name):
            station_data[station] = pickle.load(open(cache_name, "rb"))
            station_locs[station] = get_station_position(station, cache_dir=dog.cache_dir)
            continue

        station_data[station] = {'G%02d' % i: defaultdict(empty_factory) for i in range(1, 33)}
        date = start_date
        while date < start_date + duration:
            try:

                loc, data = data_for_station(dog, station, date)
                station_data[station] = station_transform(
                                            data,
                                            start_dict=station_data[station],
                                            offset=int((date - start_date).total_seconds()/30)
                                        )
                station_locs[station] = loc
            except (ValueError, DownloadError):
                print("*** error with station " + station)
            date += timedelta(days=1)
        os.makedirs("cached", exist_ok=True)
        pickle.dump(station_data[station], open(cache_name, "wb"))
    return station_locs, station_data


def get_vtec_data(dog, station_locs, station_data, conn_map=None, biases=None):
    station_vtecs = defaultdict(dict)
    def vtec_for(station, prn, conns=None, biases=None):
        if biases:
            station_bias = biases.get(station, 0)
            sat_bias = biases.get(prn, 0)
        else:
            station_bias, sat_bias = 0, 0

        if prn not in station_vtecs[station]:
            dats = []
            locs = []
            slants = []
            if station_data[station][prn]:
                end = max(station_data[station][prn].keys())
            else:
                end = 0
            for i in range(end):
                measurement = station_data[station][prn][i]
                # if conns specified, require ambiguity data
                if conns:
                    if measurement and conns[i] and conns[i].n1: # and numpy.std(conns[i].n1s) < 3:
                        res = tec.calc_vtec(
                            dog,
                            station_locs[station],
                            station_data[station][prn][i],
                            n1=conns[i].n1,
                            n2=conns[i].n2,
                            rcvr_bias=station_bias,
                            sat_bias=sat_bias,
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
                    res = tec.calc_vtec(dog, station_locs[station], station_data[station][prn][i])
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

    for station in station_data.keys():
        print(station)
        for recv in ['G%02d' % i for i in range(1, 33)]:
            if conn_map:
                vtec_for(station, recv, conns=conn_map[station][recv], biases=biases)
            else:
                vtec_for(station, recv, biases=biases)
    return station_vtecs

def correct_vtec_data(vtecs, sat_biases, station_biases):
    corrected = copy.deepcopy(vtecs)
    bad_stations = []
    for station in corrected:
        if station not in station_biases:
            bad_stations.append(station)
            continue
        for prn in ['G%02d' % i for i in range(1, 33)]:
            if prn not in corrected[station]:
                print("no sat info for %s for %s" % (station, prn))
                continue
            for i in range(len(corrected[station][prn][0])):
                dat = corrected[station][prn][0][i], corrected[station][prn][1][i], corrected[station][prn][2][i]
                if dat[0] is None:
                    continue
                dat = tec.correct_tec(dat, rcvr_bias=station_biases[station], sat_bias=sat_biases[prn])
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