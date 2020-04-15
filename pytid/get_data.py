# get observables at particular location (CORS station) at a particular time

from laika import AstroDog
from laika.gps_time import GPSTime
from laika.downloader import download_cors_station
from laika.rinex_file import RINEXFile
from laika.dgps import get_station_position
import laika.raw_gnss as raw

from collections import defaultdict
from datetime import datetime
from scipy.signal import butter, lfilter, filtfilt, sosfiltfilt
import math
import numpy


# one day worth of samples every 30 seconds
default_len = int(24*60/0.5)

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

# want to create:
# receiver -> satellite -> tick -> measurement
# from
# receiver -> tick -> measurement
def station_transform(station_data, start_dict=None, offset=0):
    if start_dict is None:
        ret = defaultdict(lambda : defaultdict(lambda : None))
    else:
        ret = start_dict

    for i, dat in enumerate(station_data):
        for sample in dat:
            # ignore non-GPS data for now
            if sample.prn.startswith("G"):
                ret[sample.prn][i + offset] = sample
    
    return ret


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
    chunked = get_contiguous(data)
    filtered_chunks = []
    for idx, chunk in chunked:
        filtered_dat = bpfilter(chunk, short_min=short_min, long_min=long_min)
        filtered_chunks.append( (idx, filtered_dat) )
    return reassemble_chunks(filtered_chunks)

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
    chunked = get_contiguous(data)
    filtered_chunks = []
    for idx, chunk in chunked:
        filtered_dat = get_depletion(chunk)
        filtered_chunks.append( (idx, filtered_dat) )
    return reassemble_chunks(filtered_chunks)