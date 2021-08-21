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
import math, random
import numpy
from multiprocessing import Pool
# import pandas as pd
import os
import pickle
import requests
from scipy.signal import butter, filtfilt
import zipfile
import ftplib, urllib, re

from laika import constants
from laika.downloader import download_cors_station, download_file
from laika.gps_time import GPSTime
from laika.lib import coordinates
from laika.rinex_file import RINEXFile, DownloadError
from laika.dgps import get_station_position
import laika.raw_gnss as raw
import laika.astro_dog

from pytid.gnss import tec, connections
from pytid.utils.configuration import missile_tid_rootfold, Configuration
import pytid.utils.thin_plate_spline as TPS
conf = Configuration()

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
    Downloader for non-CORS stations...
    TODO: Get this working...
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
        'ftp://nfs.kasi.re.kr/gps/data/daily/',
        'ftp://igs.gnsswhu.cn/pub/gps/data/daily/',
        'ftp://cddis.nasa.gov/gnss/data/daily/'
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

def data_for_station(dog_cache_dir, station_name, date):
    """
    Get data from a particular station and time. Wraps a number of laika function calls.
    Station names are CORS names (eg: 'slac')
    Dates are datetimes (eg: datetime(2020,1,7))

    The raw.read_rinex_obs(obs_data) object that is returned at the end is a list-of-lists
    of RAWGNSSMeasurement types. The outer list proceeds over the time points in the particular
    day, and the inner list proceeds over PRN (including GLONASS prns).
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
            # station_pos = get_station_position(station_name, cache_dir=dog.cache_dir)
            rinex_obs_file = download_cors_station(time, station_name, cache_dir=dog_cache_dir, leave_compressed=True)
        except (KeyError, DownloadError):
            pass

        if not rinex_obs_file:
            # station position not in CORS map, try another thing
            if station_name in extra_station_info:
                # station_pos = numpy.array(extra_station_info[station_name])
                rinex_obs_file = download_misc_igs_station(time, station_name, cache_dir=dog_cache_dir)
            else:
                raise DownloadError
    else:
        # station_pos = numpy.array(extra_station_info[station_name])
        rinex_obs_file = handlers[network](time, station_name, cache_dir=dog_cache_dir)
        
    obs_data = RINEXFile(rinex_obs_file, rate=30)
    # return station_pos, raw.read_rinex_obs(obs_data)
    return raw.read_rinex_obs(obs_data)

def meas_to_tuple(ms, stn, tick):
    '''Converts a laika RawGNSSMeasurement object to a tuple that can be neatly inserted into a structured numpy
    array.'''
    return (stn, ms.prn, tick,                                                                              #indexes
            ms.observables.get('C1C', math.nan), ms.observables.get('C2C', math.nan),
            ms.observables.get('C2P', math.nan), ms.observables.get('C5C', math.nan),
            ms.observables.get('L1C', math.nan), ms.observables.get('L2C', math.nan),
            ms.observables.get('L5C', math.nan),
            ms.recv_time_sec, ms.recv_time_week,                                                            # recv_time
            ms.sat_clock_err,                                                                               # sat_clock_err
            ms.sat_pos[0].item(), ms.sat_pos[1].item(), ms.sat_pos[2].item(),                               # sat_pos
            ms.sat_vel[0].item(), ms.sat_vel[1].item(), ms.sat_vel[2].item(),                               # sat_vel
            ms.sat_pos_final[0].item(), ms.sat_pos_final[1].item(), ms.sat_pos_final[2].item(),             # sat_pos_final
            1 if ms.corrected else 0, 1 if ms.processed else 0                                              # proc/corr status
            )

def data_for_station_npstruct_wrapper(args):
    '''
    This is a wrapper function for 'data_for_station_npstruct' that allows it to be called using a single tuple
    containing the arguments, which can be passed to multiprocessing.Pool.map(...). One important thing about this
    function though is that it also calls the 'populate_data_add_satellite_info' function at the very end. This is
    kind of an important step and probably doesn't belong in something called 'wrapper', but it seems to be working
    efficiently (I think).

    TODO: Identify whether it's a problem for 'populate_data_add_satellite_info' to be called here.

    :param args: tuple containing ( cache_dir, station_name, date_list, prn_list, start_date )
    :return:
    '''
    dog_cache_dir = args[0]
    stn_nm = args[1]
    date_list = args[2]
    prns = args[3]
    t0_date = args[4]
    date_list.sort()
    dfs_np = []

    for d in date_list:
        try:
            dfs_day = data_for_station_npstruct(dog_cache_dir, stn_nm, d, prns, t0_date)
        except:
            continue
        dfs_np.append(dfs_day)
    if len(dfs_np)==0:
        return None
    dfs=numpy.vstack(tuple(dfs_np))
    populate_data_add_satellite_info(dfs, dog_cache_dir)
    return (stn_nm, dfs)

def data_for_station_npstruct(dog_cache_dir, station_name, date, prn_list=None, t0_date=None):
    '''
    This function will wrap the one above, but instead of returning this cumbersome list-of-lists, it
    will convert it into a structured numpy array.
    :param dog_cache_dir:             (self-explanatory)
    :param station_name:    (self-explanatory)
    :param date:            (self-explanatory, start date for this particular data pull)
    :param prn_list:        a list of specific prns that should be restricted to (e.g. GPS)
    :param t0_date:         (datetime) representing the tick-0 point. For calibrating ticks.

    :return: a structured numpy array containing the results.
    '''
    time0 = datetime.now()
    t0 = GPSTime.from_datetime(date) if t0_date is None else GPSTime.from_datetime(t0_date)
    def meas_tick_calc(ms):
        return round((ms.recv_time - t0)/30.0)

    raw_obs = data_for_station(dog_cache_dir, station_name, date)
    time1 = datetime.now()

    # find out how big it needs to be:
    if prn_list is not None:
        prn_set = set(prn_list)
        tot_meas_ct = sum([len(list(filter(lambda x: x.prn in prn_set, i))) for i in raw_obs])
    else:
        tot_meas_ct = sum(list(map(len, raw_obs)))
    # Initialize the array
    dfs = numpy.empty(tot_meas_ct,
                   dtype=[('station', 'U4'), ('prn', 'U3'), ('tick', 'i4'), ('C1C', 'f8'), ('C2C', 'f8'), ('C2P', 'f8'),
                          ('C5C', 'f8'), ('L1C', 'f8'), ('L2C', 'f8'), ('L5C', 'f8'), ('recv_time_sec', 'f4'),
                          ('recv_time_week', 'i4'), ('sat_clock_err', 'f8'), ('sat_pos_x', 'f8'), ('sat_pos_y', 'f8'),
                          ('sat_pos_z', 'f8'), ('sat_vel_x', 'f8'), ('sat_vel_y', 'f8'), ('sat_vel_z', 'f8'),
                          ('sat_pos_final_x', 'f8'), ('sat_pos_final_y', 'f8'), ('sat_pos_final_z', 'f8'),
                          ('is_processed', 'i1'), ('is_corrected', 'i1')])
    # Populate it one row at a time.
    rw=0
    for i in range(len(raw_obs)):
        for j in range(len(raw_obs[i])):
            if prn_list is not None and raw_obs[i][j].prn in prn_list:
                dfs[rw] = meas_to_tuple(raw_obs[i][j], station_name, meas_tick_calc(raw_obs[i][j]))
                rw += 1
            if prn_list is None:
                dfs[rw] = meas_to_tuple(raw_obs[i][j], station_name, meas_tick_calc(raw_obs[i][j]))
                rw += 1
    time2=datetime.now()
    times=((time1-time0).seconds+round((time1-time0).microseconds/1000000,2), (time2-time1).seconds+round((time2-time1).microseconds/1000000,2))
    return dfs.reshape((tot_meas_ct,1))

def get_dates_in_range(start_dt, durr):
    '''returns a list of datetime objects separated by a day that fall in the range of start_dat and durration.'''
    dates = []
    date_var = start_dt
    while date_var < start_dt + durr:
        dates.append(date_var)
        date_var += timedelta(days=1)
    return dates

def cors_get_station_lists_for_day(dt):
    '''For a particular day, pulls a list of CORS stations that have data posted for each of the two CORS sites.
    Technically it just returns a directory listing limited only to items of length 4, but this will do.'''
    url1='https://geodesy.noaa.gov/corsdata/rinex/'
    print('getting cors station list for %s' % dt)
    day_folder = url1 + dt.strftime('%Y/%j/')
    with urllib.request.urlopen(day_folder) as response:
        html = response.read().decode('utf-8')
    #
    # ...this pattern worked when I checked the site on 8/20/21...
    pat = '<a href="..../">(?P<st>....)/</a>'
    prog = re.compile(pat)
    res = prog.finditer(html)
    stn_list = []
    for r in res:
        stn_list.append(r.group('st'))

    return stn_list, stn_list


# def cors_get_station_lists_for_day_OLD(dt):
#     '''DEPRECATED: NOAA HAS SHIFTED TO HTTP...sigh
#     For a particular day, pulls a list of CORS stations that have data posted for each of the two CORS sites.
#     Technically it just returns a directory listing limited only to items of length 4, but this will do.'''
#     # f1 = 'ftp://geodesy.noaa.gov/cors/rinex/'
#     # f2 = 'ftp://alt.ngs.noaa.gov/cors/rinex/'
#     print('getting cors station list for %s' % dt)
#     f1='geodesy.noaa.gov'
#     f2='alt.ngs.noaa.gov'
#     day_folder = "/corsdata/rinex/" +  dt.strftime('%Y/%j/')
#     ftp = ftplib.FTP(f1, "anonymous", "")
#     f1nlst = ftp.nlst(day_folder)
#     ftp.quit()
#     ftp = ftplib.FTP(f2, "anonymous", "")
#     f2nlst = ftp.nlst(day_folder)
#     ftp.quit()
#     f1sta = list(filter(lambda x: len(x)==4, list(map(lambda x: x.replace(day_folder,''), f1nlst))))
#     f2sta = list(filter(lambda x: len(x) == 4, list(map(lambda x: x.replace(day_folder, ''), f2nlst))))
#     return f1sta, f2sta

def cors_download_commands(stns, dt_list, out_script='bash_cors_obs_download.sh', cache_dir='~/.gnss_cache'):
    '''
    This function takes a set of stations and a set of dates and it generates a bash script to download all
    of the necessary cors_obs files into the correct place in the '.gnss_cache' folder and properly unzip each one.
    Importantly, it only does this for the stations that are present in the cors_obs ftp sites for the particular
    day. The bash script is located in the missile-tid/scripts folder.

    When the script is run, it processes the downloads correctly and runs in a fraction of the time of the laika
    processes. Laika downloads are quite slow, so that is not surprising. Once the files are in the right place,
    existing processes can recognize that they dont need to be re-downloaded and we can move on with life.

    :param stns:        list of stations to download for
    :param dt_list:     list of dates on which to donwload it
    :param out_script: name of the file for the bash script. It will natrually be thrown in the 'scripts' folder.
    :param cache_dir: the one from AstroDog
    :return:
    '''
    def get_dt_cors_stns(dt):
        '''Helper function to ID the subset of <stns> that will have CORS files to download.'''
        cors_stns = cors_get_station_lists_for_day(dt)
        my_cors_stns_tup = (set(stns).intersection(list(cors_stns[0])), set(stns).intersection(list(cors_stns[1])))
        my_cors_stns = list(set(my_cors_stns_tup[0]).union(set(my_cors_stns_tup[1])))
        return my_cors_stns

    url_base1='https://geodesy.noaa.gov/corsdata/rinex/'
    # url_base2='ftp://alt.ngs.noaa.gov'
    bash_script = open(os.path.join(missile_tid_rootfold, 'scripts', out_script), 'w')
    if cache_dir[0]=='~':
        cache_dir = os.path.expanduser(cache_dir)

    def cors_url_to_bash(stn, tgt_date):
        '''Changing this to not to the unzip command because we are going to start reading files directly as .gz'''
        filename1 = stn + tgt_date.strftime("%j0.%yo.gz")
        filename2 = stn + tgt_date.strftime("%j0.%yd.gz")
        cors_file_url1 = url_base1 + tgt_date.strftime('%Y/%j/') + stn + '/' + filename1
        cors_file_url2 = url_base1 + tgt_date.strftime('%Y/%j/') + stn + '/' + filename2
        dl_target_fold = os.path.join(cache_dir, 'cors_obs', tgt_date.strftime('%Y'), tgt_date.strftime('%j'), stn)
        l1ct = bash_script.write('wget %s -P %s -nc \n' % (cors_file_url1, dl_target_fold))
        l2ct = bash_script.write('if [ $? -eq 0 ]; then echo %s; \n' % os.path.join(dl_target_fold, filename))
        l3ct = bash_script.write('else wget %s -P %s -nc ; \n' % (cors_file_url2 , dl_target_fold))
        l4ct = bash_script.write('if [ $? -eq 0 ]; then echo %s; fi; fi;\n\n' % os.path.join(dl_target_fold, filename))

    for my_dt in dt_list:
        date_cors_stns = get_dt_cors_stns(my_dt)
        print('%s - ' % my_dt, end='\r', flush=True)
        for s in date_cors_stns:
            print('%s - %s' % (my_dt,s), end='\r', flush=True)
            cors_url_to_bash(s, my_dt)
    bash_script.close()

def get_ftp_folder_ls(site, folder):
    '''Pings an FTP folder and lists all the files in it.'''
    ftp = ftplib.FTP(site, "anonymous","")
    flist = ftp.nlst(folder)
    ftp.quit()
    return flist

def get_igs_station_lists_for_day(dt):
    '''The idea for this is to write two functions similar to the previous two, but that can
    handle IGS stations instead of strictly CORS. This will be a bit more involved obviously.
    NOTE: THIS IS NOT CURRENTLY IMPLEMENTED.'''
    # TODO: Complete this
    f1=('garner.ucsd.edu' , '/archive/garner/rinex/')
    f2=('data-out.unavco.org','/pub/rinex/obs/')
    f3=('nfs.kasi.re.kr','/gps/data/daily/')
    f4=('igs.gnsswhu.cn','/pub/gps/data/daily/')
    f5=('cddis.nasa.gov','/gnss/data/daily/')
    Y_str = dt.strftime('%Y')
    y_str = dt.strftime('%y')
    j_str = dt.strftime('%j')

    #site 1: UCSD
    day_folder = f1[1] + dt.strftime('%Y/%j/')
    # file_list = get_ftp_folder_ls(f1[0],day_folder)
    myre = re.compile('[0-9a-zA-Z]{4}' + j_str + '0.' + y_str + 'o.Z')
    # Note: UCSD currently only hosts the high-rate hatanaka rinex files (i.e. '*d.Z' files). Laika currently doesn't
    #   support those, so the site 1 list is not implemented here.

    #site 2: Unavco
    day_folder = f2[1] + dt.strftime('%Y/%j/')
    file_list = get_ftp_folder_ls(f2[0], day_folder)
    myre = re.compile('[0-9a-zA-Z]{4}' + j_str + '0.' + y_str + 'o.Z')


    #TODO: Finish implementing this
    pass

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

    def __init__(self, dog, start_date, duration, stations, prn_list = None, data_struct='dict'):
        '''data_struct must be one of ['dict','dense']'''
        self.dog = dog
        self.dog_cache_dir = dog.cache_dir
        self.start_date = start_date
        self.duration = duration
        self.date_list = get_dates_in_range(self.start_date, self.duration)
        self.stations = stations
        self._station_locs = None
        self._station_data = None
        self._clock_biases = None
        self.cache_file_prefix = None
        # Local coordinates for the stations we're using
        # this enables faster elevation lookups
        self.station_converters = dict()
        self.prn_list = prn_list #if we want to restrict the set of available satellites
        self.station_data_structure = data_struct
        self.station_data_sources = None

    @property
    def station_locs(self):
        if self._station_locs is None:
            self.populate_data(method=self.station_data_structure)
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

    def populate_station_locs(self):
        '''Separates this step from the populate_data step. They didn't really need to be together.'''
        for station in self.stations:
            try:
                self._station_locs[station] = get_station_position(station, cache_dir=self.dog.cache_dir)
            except KeyError:
                self._station_locs[station] = numpy.array(extra_station_info[station])

    def prepare_data_sources(self, cache_file_prefix):
        '''This does some of the legwork up front to see which stations' data can be recruited from where. Right
        now this just tests for whether it is in the cache folder, and then failing that whether a folder for
        that station is posted to either of the cors FTP sites on taht particular day. If there is at least
        one on every day in the range, then that is the plan.

        TODO: Add a round to check the misc_igs websites ater checking the cors sites.'''
        self.date_list = get_dates_in_range(self.start_date, self.duration)
        self.station_data_sources = dict.fromkeys(self.stations)
        print('preparing data sources')

        # First, run through each station to see if we have that pickle file
        pickle_file_ct = 0
        self.cache_file_prefix = cache_file_prefix
        for stn in self.stations:
            # cache_name = "cached/%s_%s_%s_to_%s.p" % ( cache_file_prefix, stn,
            cache_name = "%s_%s_%s_to_%s.p" % (cache_file_prefix, stn,
                self.start_date.strftime("%Y-%m-%d"), (self.start_date + self.duration).strftime("%Y-%m-%d"))
            cache_name = os.path.join(conf.missile_tid_cache, 'stationdat', cache_name)
            # print('cache_name: %s, exists=%s' % (cache_name, os.path.exists(cache_name)))
            if os.path.exists(cache_name):
                self.station_data_sources[stn]=('cached_pickle', cache_name)
                pickle_file_ct += 1

        # Then pull down the CORS lists for each day:
        cors_lists = dict.fromkeys(range(len(self.date_list)))
        for i in range(len(self.date_list)):
            cors_lists_d = cors_get_station_lists_for_day(self.date_list[i])
            cors_lists[i] = cors_lists_d
        cors_ftp_ct = 0
        for stn in self.stations:
            if self.station_data_sources[stn] is None:
                stn_daily_options = []
                stn_cors_ftp_ok_alldays = True
                stn_cors_ftp_ok_onedays = False
                for i in range(len(self.date_list)):
                    in_f1 = stn in cors_lists[i][0]; in_f2 = stn in cors_lists[i][1]
                    if (not in_f1) and (not in_f2):
                        stn_cors_ftp_ok_alldays = False
                    if in_f1 or in_f2:
                        stn_cors_ftp_ok_onedays = True
                    stn_daily_options.append((in_f1, in_f2))
                if stn_cors_ftp_ok_onedays:
                    self.station_data_sources[stn] = ('cors_ftp', stn_daily_options)
                    cors_ftp_ct += 1

        print('%s total stns, %s are pickled, %s are on CORS ftp sites.' % (len(self.station_data_sources),
                                                                            pickle_file_ct, cors_ftp_ct))

    def populate_data(self):
        '''
        wrapper for two methods to populate the data now.
        :return:
        '''
        self._station_locs = {}
        self._station_data = {}

        self.bad_stations = []
        self.populate_station_locs()

        for station in self.stations:
            print(station)
            cache_name = "cached/stationdat_%s_%s_to_%s.p" % (
                station,
                self.start_date.strftime("%Y-%m-%d"),
                (self.start_date + self.duration).strftime("%Y-%m-%d")
            )
            # Check to see if we have cached this thing. If so, recover it from there
            if os.path.exists(cache_name):
                self._station_data[station] = pickle.load(open(cache_name, "rb"))
                continue

            # Start populating this dictionary.
            self._station_data[station] = {prn: defaultdict(empty_factory) for prn in satellites}
            date = self.start_date
            while date < self.start_date + self.duration:
                try:
                    # loc, data = data_for_station(self.dog, station, date)
                    data = data_for_station(self.dog, station, date)
                    self._station_data[station] = station_transform(
                        data,
                        start_dict=self.station_data[station],
                        offset=int((date - self.start_date).total_seconds() / 30)
                    )
                    # self.station_locs[station] = loc
                except (ValueError, DownloadError):
                    print("*** error with station " + station)
                    self.bad_stations.append(station)
                date += timedelta(days=1)
            os.makedirs("cached", exist_ok=True)
            pickle.dump(self._station_data[station], open(cache_name, "wb"))

        for bad_station in self.bad_stations:
            self.stations.remove(bad_station)

    # def populate_data_dict(self):
    #     self._station_data = {}
    #     for station in self.stations:
    #         print(station)
    #         cache_name = "cached/stationdat_%s_%s_to_%s.p" % (
    #             station,
    #             self.start_date.strftime("%Y-%m-%d"),
    #             (self.start_date + self.duration).strftime("%Y-%m-%d")
    #         )
    #         # Check to see if we have cached this thing. If so, recover it from there
    #         if os.path.exists(cache_name):
    #             self._station_data[station] = pickle.load(open(cache_name, "rb"))
    #             # try:
    #             #     self.station_locs[station] = get_station_position(station, cache_dir=self.dog.cache_dir)
    #             # except KeyError:
    #             #     self.station_locs[station] = numpy.array(extra_station_info[station])
    #             continue
    #
    #         # Start populating this dictionary.
    #         self._station_data[station] = {prn: defaultdict(empty_factory) for prn in satellites}
    #         date = self.start_date
    #         while date < self.start_date + self.duration:
    #             try:
    #                 # loc, data = data_for_station(self.dog, station, date)
    #                 data = data_for_station(self.dog, station, date)
    #                 self._station_data[station] = station_transform(
    #                                             data,
    #                                             start_dict=self.station_data[station],
    #                                             offset=int((date - self.start_date).total_seconds()/30)
    #                                         )
    #                 # self.station_locs[station] = loc
    #             except (ValueError, DownloadError):
    #                 print("*** error with station " + station)
    #                 self.bad_stations.append(station)
    #             date += timedelta(days=1)
    #         os.makedirs("cached", exist_ok=True)
    #         pickle.dump(self._station_data[station], open(cache_name, "wb"))

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

    def get_measure(self, sta, prn, tick, **kwargs):
        '''Once the data is populated, a unified method to get either a laika.GNSSmeasurement object (in the old
        structure, or a row of the structured numpy array in the dense structure, or a row of the structured vtec
        numpy array in the dense structure..'''

        if not self.check_sta_prn_tick_exist(sta, prn, tick): # 1) if the tick doesn't exist, return None
            return None
        if prn not in self._station_data[sta] or tick not in self._station_data[sta][prn]:
            return None
        else:
            return self._station_data[sta][prn][tick]

    def check_sta_prn_tick_exist(self, sta, prn, tick=None):
        '''Returns True if there is a tick for the particular prn/station. If tick is omitted, returns true if
        the PRN exists for the station.'''
        if tick is None:
            return prn in self._station_data[sta]
        else:
            return prn in self._station_data[sta] and tick in self._station_data[sta][prn]

    def get_tick_list_for_prn(self, sta, prn):
        '''produces a list of valid ticks for a given station-prn combo.'''
        return list(self._station_data[sta][prn].keys())

    def get_prn_list_for_station(self, sta):
        '''returns a list of prns present for a given station.'''
        return list(self._station_data[sta].keys())


class ScenarioInfoDense(ScenarioInfo):

    def __init__(self, dog, start_date, duration, stations, prn_list = None):
        ScenarioInfo.__init__(self, dog, start_date, duration, stations, prn_list=prn_list, data_struct='dense')
        self._row_by_prn_tick_index = None
        self.station_vtecs = None
        self.conns = None
        self.conn_map = None
        self.bias_repo = {}

    def populate_data(self, parallel_proc=True):
        '''
        In this version, the outermost object is a pandas data-frame instead of being a nested-dict.
        :return:
        '''
        self._station_locs = {}
        self._station_data = {}

        self.bad_stations = []
        self.populate_station_locs()
        self.prepare_data_sources('stationdat_dense') #'stationdat_dense' is the cache_file_prefix

        # Make list of pickled data sources (pickled_stns) versus download list (cors_stns).
        cors_stns = []; pickle_stns=[]; none_stns=[];
        for cs in self.stations:
            if self.station_data_sources[cs] is None:
                none_stns.append(cs)
                self.bad_stations.append(cs)
            elif self.station_data_sources[cs][0]=='cors_ftp':
                cors_stns.append(cs)
            elif self.station_data_sources[cs][0]=='cached_pickle':
                pickle_stns.append(cs)
        print('Station Data Sources: Cached=%s, ftp=%s, None=%s' % (len(pickle_stns), len(cors_stns), len(none_stns)))
        # Take care of any stations that ended up in the none_stns list:
        if len(none_stns)>0:
            print('   \'None\' stations: %s' % str(none_stns))

        # *** STEP 1 ***: Get Pickled Files and read them into self._station_data:
        print('Getting pickled files....')
        for station in pickle_stns:
            print('  Station: %s (%s of %s) ' % (station, pickle_stns.index(station), len(pickle_stns)), end = '\r')
            if self.station_data_sources[station][0]=='cached_pickle':
                cache_name=self.station_data_sources[station][1]
                if os.path.exists(cache_name):
                    self._station_data[station] = pickle.load(open(cache_name, "rb"))
        print(' '*50)

        # *** STEP 2a ***: Prepare to download the stations in 'cors_stns' in parallel:
        print('Getting files from FTP....')
        args_list = []
        numpy.seterr(invalid='ignore')
        # for every station we're going to make a tuple of the form:
        #       ( <AstroDog.cache_dir> , station, date_list, prn_list, start_date )
        for station in cors_stns:
            # my_dl is a list of dates for which the particular station is available for FTP download
            my_dl=[self.date_list[i] for i in range(len(self.date_list)) if (self.station_data_sources[station][1][i][0] or self.station_data_sources[station][1][i][1])]
            args_list.append((self.dog.cache_dir, station, my_dl, self.prn_list.copy(), self.start_date))

        # *** STEP 2b ***: Using Multiprocessing to run the 'data_for_station' wrapper function
        #       ...also adding satellite info while we're at it.
        args_list_subs = [args_list[i::10] for i in range(10)]
        total_done = 0; tstart=datetime.now();
        if parallel_proc:
            p = Pool(int(os.cpu_count() / 2))
            for args_list_mini in args_list_subs:
                if len(args_list_mini)==0:
                    continue
                data_read = p.map(data_for_station_npstruct_wrapper, args_list_mini)
                total_done+= len(data_read)
                for i in range(len(data_read)):
                    if data_read[i][1] is not None:
                        self._station_data[data_read[i][0]]=data_read[i][1]
                print('%s of %s done (%s elapsed)' % (total_done, len(args_list), datetime.now()-tstart))

                #Pickle them while we're here:
                for i in range(len(data_read)):
                    if data_read[i][1] is None:
                        continue
                    cache_file_name = "stationdat_dense_%s_%s_to_%s.p" % (data_read[i][0], self.start_date.strftime("%Y-%m-%d"),
                                                                          (self.start_date + self.duration).strftime(
                                                                              "%Y-%m-%d"))
                    cache_path = os.path.join(missile_tid_rootfold, 'cached', cache_file_name)
                    os.makedirs("cached", exist_ok=True)
                    with open(cache_path, 'wb') as tempcache:
                        print('pickling %s' % data_read[i][0], end='\r')
                        pickle.dump(self._station_data[data_read[i][0]], tempcache)
            p.close()
        else:
            # no multiprocessing: do them one at a time (womp, womp)
            for arg in args_list:
                one_data_read = data_for_station_npstruct_wrapper(arg)
                self._station_data[one_data_read[0]]=one_data_read[1]
                total_done += 1
                print('%s of %s done (%s elapsed)' % (total_done, len(args_list), datetime.now() - tstart))
                #pickle it
                cache_file_name = "stationdat_dense_%s_%s_to_%s.p" % (one_data_read[i][0], self.start_date.strftime("%Y-%m-%d"),
                                                                      (self.start_date + self.duration).strftime("%Y-%m-%d"))
                cache_path = os.path.join(missile_tid_rootfold, 'cached', cache_file_name)
                os.makedirs("cached", exist_ok=True)
                with open(cache_path, 'wb') as tempcache:
                    print('pickling %s' % station, '\r')
                    pickle.dump(self._station_data[station], tempcache)

        # wrap up:
        print('Finishing data population process...')
        numpy.seterr(invalid='warn')
        nones=[k for k in self.station_data.keys() if self._station_data[k] is None]
        for n in nones:
            self._station_data.pop(n)
            self.stations.remove(n)
        for bad_station in self.bad_stations:
            self.stations.remove(bad_station)

        # Finally, run the QC on each matrix:
        observables_qc_fail = [k for k in self.station_data.keys() if quality_check_station_obs(self._station_data[k]) == False]

        print('Removing %d stations from the data due to observables QC failure...' % len(observables_qc_fail))
        self.observables_qc_fail_stations = observables_qc_fail
        for qc_fail_station in observables_qc_fail:
            self._station_data.pop(qc_fail_station)
            self.stations.remove(qc_fail_station)

        # Make the station-prn-index lookup:
        self.populate_data_index_row_by_prn_tick()


    @property
    def row_by_prn_tick_index(self):
        if self._row_by_prn_tick_index is None:
            self.populate_data_index_row_by_prn_tick()
        return self._row_by_prn_tick_index

    def populate_data_index_row_by_prn_tick(self):
        '''Creates a nested dictionary that returns the row of the nparray given a station/PRN/tick combo. Will use
        a different function to use this to return the measurement directly.'''
        self._row_by_prn_tick_index = dict.fromkeys(self._station_data.keys())
        prn_set = set([])
        # iterate over stations
        print('Making data prn-tick to row index...')
        ct = 0
        for stn in self._row_by_prn_tick_index.keys():
            ct += 1
            print('  %s (%s of %s)' % (stn, ct, len(self._station_data.keys())) , end = '\r')
            prn_list = list(numpy.unique(self._station_data[stn]['prn']))
            prn_set |= set(prn_list)
            self._row_by_prn_tick_index[stn] = dict.fromkeys(prn_list)
            # iterate over satellites
            for prn in self._row_by_prn_tick_index[stn].keys():
                prn_rows = numpy.where(self._station_data[stn]['prn']==prn)[0]
                prn_ticks = self._station_data[stn]['tick'][prn_rows][:,0]
                self._row_by_prn_tick_index[stn][prn] = dict(zip(prn_ticks, prn_rows))
        self.prn_list = list(prn_set)
        self.prn_list.sort()
        print('  done', end='\r'); print('');

    def populate_data_index_from_pickled(self):
        '''Gives the option to read the row index from a cached file. Should probably verify it is the right one but
        that is for later.'''
        dedicated_row_index_file_loc = os.path.join(conf.missile_tid_root, 'cached', 'conns','row_by_prn_tick_index.p')
        with open(dedicated_row_index_file_loc, 'rb') as rif:
            self._row_by_prn_tick_index = pickle.load(rif)

    def make_dense_station_vtecs(self):
        '''Setting up the station_vtecs matrix which is going to correspond to the station_data arrangement'''
        self.station_vtecs = dict.fromkeys(self.stations)
        for stn in self.station_vtecs.keys():
            stn_obs_ct = self._station_data[stn].shape[0]
            self.station_vtecs[stn] = numpy.empty(stn_obs_ct, dtype=[
                ('raw_vtec', 'f8'), ('ion_loc_x', 'f8'),('ion_loc_y', 'f8'), ('ion_loc_z', 'f8'),
                ('s_to_v', 'f8'), ('corr_vtec', 'f8'), ('is_bias_corrected','i1')])

    def gather_connections(self, save_to_cache=True, check_cache_to_load=True):
        '''
        Does the computing of connections. It is designed right now to break up the stations into small groups
        and save the results along the way in the cached folder. Otherwise this takes way too long to run.
            -- On 1700 stations this takes several hours to run.
            -- It is also set up to try and read from the cache files if they exist

        Method: if there are more than 200 stations, it will cut the stations in to exactly 50 subgroups, each
        of which gets its own file.

        TODO: 1) Make this a bit smarter about choosing when to cache and when not to.
        TODO: 2) Clean up some of the partitioned pickle files once it's done running.
        TODO: 3) Implement some parallelism to speed the damn thing up...

        Parameters
        ----------
        save_to_cache : bool
        check_cache_to_load : bool

        Returns
        -------

        '''
        conns_cache = os.path.join(missile_tid_rootfold, 'cached', 'conns')
        if save_to_cache:
            os.makedirs(conns_cache, exist_ok=True)
        if len(self.stations) > 200:
            # Station list broken into exactly 50 groups.
            station_groups = [self.stations[i::50] for i in range(50)] #50 groups
        else:
            station_groups = [self.stations,]
        self.conns = []

        # As it is 'gathering' the connections for its set of stations, check to
        #   see if that file has already been done. Read it in if so.
        for i in range(len(station_groups)):
            print("Station Group %s, (%s stations)" % (i, len(station_groups[i])))
            sg = station_groups[i]
            conns_cache_fn = os.path.join(conns_cache, 'conns_stationgroup_%s.p' % i)
            if check_cache_to_load and os.path.exists(conns_cache_fn):
                # Cache exists, read it in.
                print('opening group %s from pickled file %s' % (i, conns_cache_fn), end='\r')
                with open(conns_cache_fn, 'rb') as ccf:
                    cns = pickle.load(ccf)
            else:
                # No such luck, go to the trouble from scratch
                cns = connections.get_connections(self, station_subset=sg)
                # One the new group has been calculated, pickle it immediately so it doesn't get lost
                with open(conns_cache_fn,'wb') as ccf:
                    pickle.dump(cns, ccf)
                    print('pickled connection group %s' % i, end = '\r')
            self.conns += cns

    def connections_save_to_cache(self):
        pass

    def adjust_connections(self):
        '''
        Two final steps:
            1) Final cycle-slip clean-up via change-point detection
            2) Make the connection-map object
        Returns : None
        -------
        '''
        print('Final check to clean up remaining cycle-slips...', end = '', flush=True)
        connections.find_and_remove_remaining_cycle_slips(self)
        print('done.\n', flush=True)

        print('Making connections map...', end ='', flush=True)
        self.conn_map = connections.make_conn_map(self.conns)
        print('done.', flush=True);

    def reset_connections_map(self):
        '''Calls the make_conn_map() real quick in case of a re-load, just to keep them synced.'''
        if self.conn_map is not None:
            del self.conn_map
            self.conn_map = None
        print('Making connections map...', end='', flush=True)
        self.conn_map = connections.make_conn_map(self.conns)
        print('done.', flush=True);

    def resolve_ambiguities(self, use_offset_method = False):
        '''
        Runs the conns_correct_* method to estimate the ambiguities, either via computing the offset value
            or using the direct estimation with least-squares.

        Parameters
        ----------
        use_offset_method : bool
            If True, will calculate the offset value rather than .n1 and .n2 in the Connection
        -------
        '''
        if use_offset_method:
            print('Running ambiguity correction using offset method...', end='')
            connections.correct_conns_code(self, self.conns)
        else:
            print('Running ambiguity correction using least squares...', end='')
            connections.correct_conns(self, self.conns)
        print('done.')

    def correct_vtec_data_dense(self, bias_dict):
        '''Runs down the station_vtecs object and puts in the right value for `corrected_vtec`'''
        # TODO: add something to handle glonass
        bad_stations = [];
        # Iterate over stations
        for stn in self.station_vtecs.keys():
            if stn not in bias_dict:
                bad_stations.append(stn)
                continue
            else:
                stn_bias = bias_dict[stn]
            # Iterate over PRNs with data for that station
            for prn in self.get_prn_list_for_station(stn):
                if prn not in bias_dict:
                    continue
                else:
                    # TODO: HANDLE GLONASS HERE
                    prn_bias = bias_dict[prn]
                tick_row_lkp = self.row_by_prn_tick_index[stn][prn]
                # Iterate over rows in of that PRN:
                for t, r in tick_row_lkp.items():
                    vt = self.station_vtecs[stn][r]
                    if not numpy.isnan(vt['raw_vtec']):
                        self.station_vtecs[stn][r]['corr_vtec'] = tec.correct_tec_vals(vt['raw_vtec'], vt['s_to_v'],
                                                                                       stn_bias, prn_bias)
        for station in bad_stations:
            print("missing bias data for %s: deleting vtecs" % station)

    def check_sta_prn_tick_exist(self, sta, prn, tick=None):
        '''Returns True if there is a tick for the particular prn/station. If tick is omitted, returns true if
        the PRN exists for the station.'''
        if tick is None:
            return prn in self.row_by_prn_tick_index[sta]
        else:
            return prn in self.row_by_prn_tick_index[sta] and tick in self.row_by_prn_tick_index[sta][prn]

    def get_measure(self, sta, prn, tick, row_only=False, vtec=False):
        '''Once the data is populated, a unified method to get either a laika.GNSSmeasurement object (in the old
        structure, or a row of the structured numpy array in the dense structure, or a row of the structured vtec
        numpy array in the dense structure..'''

        if not self.check_sta_prn_tick_exist(sta, prn, tick):
            # 1) if the tick doesn't exist, return None
            return None
        if not row_only and not vtec:
            # 2) Return the np-measurement
            return self._station_data[sta][self.row_by_prn_tick_index[sta][prn][tick]]
        elif row_only:
            # 3) Return the row index only
            return self.row_by_prn_tick_index[sta][prn][tick]
        else:
            # 4) Return the vtec np-measurement
            return self.station_vtecs[sta][self.row_by_prn_tick_index[sta][prn][tick]]

    def get_tick_list_for_prn(self, sta, prn):
        '''produces a list of valid ticks for a given station-prn combo.'''
        return list(self.row_by_prn_tick_index[sta][prn].keys())

    def get_prn_list_for_station(self, sta):
        return list(self.row_by_prn_tick_index[sta].keys())

    def get_vtec_data(self, load_from_cache = False):
        '''Has the option to load it from a pickle. Should add something to verify it is the right stations,
        dates etc...

        This method takes quite a while to run...
        '''
        if load_from_cache:
            cache_station_vtecs = os.path.join(cache_conns_folder, 'station_vtecs_offset.p')
            if os.path.exists(cache_station_vtecs):
                with open(cache_station_vtecs, 'rb') as cca:
                    scenario_test.station_vtecs = pickle.load( cca)
        elif self.station_vtecs is None:
            # 1) Creates the empty numpy array with same shape as main observations.
            self.make_dense_station_vtecs()
            # 2) Goes through and processes the calculations.
            get_vtec_data_dense(self, self.conn_map)

    def get_bias_for_day(self, day_index, group_sep_hrs=2, knot_sep_mins=5, max_stations_per_calc=350, rseed=1111,
                         min_number_calc_iters=None):
        '''
        For a particular day in the sequence held by the scenario, compute the set of biases for the stations and
        satellites in the data. If the number of stations exceeds the max, spread it out a bit and record all the
        results.

        TODO: Fix this method and re-test it. Probably have to fix something upstream first.

        :param day_index: Which day in the day_list to compute them for
        :param group_sep_hrs: How many hours apart to take the groups (default 2)
        :param knot_sep_mins: How many miniutes apart to separate the knots from the un-knots (default 5)
        :param max_stations_per_calc: Maximum number of stations to involve (to limit data size, limit 300)
        :return: Eventually adds to the dictionary bias_repo which looks like this:
            bias_repo = { 'YYYY-MM-DD' : {  'G01' :  { 1: 2.15, 2: ...(bias calcs)...},
                                            'G02' :  { 1: ....(bias calcs)... },
                                            ....
                                            'station1': { <group#> : <bias>},...
                                        }
                          'YYYY-MM-D2': { .... }, ....
                          }
        '''
        assert day_index < self.duration.days
        if not isinstance(day_index, int):
            day_index = math.floor(day_index)
        num_groups = math.floor(24.0/group_sep_hrs)
        group_sep_ticks = group_sep_hrs * 120
        knot_sep_ticks = knot_sep_mins * 2
        first_tick = day_index * 2 * 60 * 60 * 24

        # Number of separate bias calcs we'll have to do to get data for all the stations. (Break the list into equal
        #   parts)
        station_list = list(self.station_vtecs.keys()); station_list.sort();
        n_calc_iters = math.ceil(len(station_list)/max_stations_per_calc)
        stations_per_calc = math.ceil(len(station_list)/n_calc_iters)
        random.seed(rseed); random.shuffle(station_list);
        n_prns = len(self.prn_list)
        station_subgroups = [station_list[i::n_calc_iters] for i in range(n_calc_iters)]
        if min_number_calc_iters is not None and n_calc_iters < min_number_calc_iters:
            for i in range(min_number_calc_iters-n_calc_iters):
                sts = random.sample(station_list, stations_per_calc)
                station_subgroups.append(sts)
            n_calc_iters = min_number_calc_iters

        day_biases = {k: {} for k in  (self.prn_list+station_list)}
        self.bias_QC_report = {}
        self.bias_QC_report['PRN_bias_table'] = numpy.zeros((n_prns,n_calc_iters), dtype=numpy.float64)
        self.bias_QC_report['PRN_labels'] = self.prn_list.copy()
        self.bias_QC_report['station_subgroups'] = station_subgroups.copy()
        self.bias_QC_report['day_index'] = day_index
        self.bias_QC_report['date'] = self.date_list[day_index]
        self.bias_QC_report['run_time'] = datetime.now()
        self.bias_QC_report['num_sub_calcs'] = n_calc_iters
        self.bias_QC_report['stations_per_subcalc'] = stations_per_calc


        for i in range(n_calc_iters):
            print("Computing biases for day %s, group %s, %s stations, at %s..." %
                  (day_index, i, len(station_subgroups[i]), datetime.now()), end=''); t1=datetime.now();
            b,z,info,biases = TPS.bias_multi_tps_solve(self,first_tick, [0,], [knot_sep_ticks,], group_sep_ticks, num_groups=num_groups,
                                     prns=self.prn_list, stns = station_subgroups[i], use_sparse=False)
            print("done (%s)" % (datetime.now()-t1))
            for k,v in biases.items():
                day_biases[k][i]=v
                if k in self.prn_list:
                    rw = self.prn_list.index(k)
                    self.bias_QC_report['PRN_bias_table'][rw,i] = v

        self.bias_repo[(self.start_date + timedelta(days=day_index)).strftime('%Y-%m-%d')] = day_biases
        self.bias_QC_report['PRN_bias_mu_std'] = numpy.vstack((numpy.mean(self.bias_QC_report['PRN_bias_table'],axis=1),
                                                            (numpy.var(self.bias_QC_report['PRN_bias_table'],axis=1)*(n_calc_iters/(n_calc_iters-1)))**.5)).transpose()
        mymean = lambda x: sum(x)/len(x)
        self.bias_est_avg = {k: mymean(day_biases[k].values()) for k in day_biases.keys()}

    def save_bias_repo(self, clobber=False):
        '''Method to save the current version of the bias repository.'''
        if not clobber:
            myf = open(os.path.join(missile_tid_rootfold, 'data','bias_calc_database.csv'), 'a')
        else:
            myf = open(os.path.join(missile_tid_rootfold, 'data','bias_calc_database.csv'), 'w')
            cct=myf.write('date,stn_prn,group,bias,date_run\n')
        curr_time = datetime.now()
        for dt in self.bias_repo.keys():
            for stnprn in self.bias_repo[dt].keys():
                for grp, bias_calc in self.bias_repo[dt][stnprn].items():
                    cct=myf.write('%s,%s,%s,%s,%s\n' % (dt, stnprn, grp, bias_calc, curr_time))
        myf.close()

    def write_bias_QC_report(self, outpath=None):
        '''Writes the bias QC info to a file where it acts as a report of the stability'''
        if self.bias_QC_report is None:
            return
        k = self.bias_QC_report['num_sub_calcs'];
        n = self.bias_QC_report['stations_per_subcalc'];
        n_prns = len(self.bias_QC_report['PRN_labels']); prns = self.bias_QC_report['PRN_labels'];
        bias_mu_std = self.bias_QC_report['PRN_bias_mu_std']
        if outpath is None:
            datestr = self.bias_QC_report['date'].strftime('%Y%m%d')
            outpath = os.path.join(conf.missile_tid_cache,'bias_QC_%s_%diters_%dstns.txt' % (datestr,k,n))

        rpt = open(outpath,'w')
        cct = rpt.write('date:\t%s\n' % self.bias_QC_report['date'])
        cct = rpt.write('day_index:\t%s\n' % self.bias_QC_report['day_index'])
        cct = rpt.write('run_time:\t%s\n' % self.bias_QC_report['run_time'])
        cct = rpt.write('num_sub_calcs:\t%s\n' % self.bias_QC_report['num_sub_calcs'])
        cct = rpt.write('stations_per_subcalc:\t%s\n' % self.bias_QC_report['stations_per_subcalc'])
        cct = rpt.write('\nBias Estimates\n')
        cct = rpt.write('PRN,' + ','.join(map(str,range(k))) + ',,Mean,StdErr\n')
        blank_line = '%s,' + ','.join(['%f',]*k) + ',,%f,%f\n'
        for r in range(n_prns):
            bvals=self.bias_QC_report['PRN_bias_table'][r,:]
            ents=(prns[r],) + tuple(bvals) + ('',bias_mu_std[r,0], bias_mu_std[r,1])
            cct = rpt.write(blank_line % ents)

        cct = rpt.write('\nStation Groupings:\n')
        maxstns=max(map(len, self.bias_QC_report['station_subgroups']))
        stn_grps = []
        for i in range(k):
            stn_grps.append(self.bias_QC_report['station_subgroups'][i] + ['',]*(maxstns-len(self.bias_QC_report['station_subgroups'][i])))
        stn_blank = ','.join(['%s',]*5) + '\n'
        for i in range(maxstns):
            cct = rpt.write(stn_blank % tuple(map(lambda x: stn_grps[x][i], range(k))))

        rpt.close()




def quality_check_station_obs(station_matrix, max_absolute_n21_val = 5000):
    '''
    Runs some checks to make sure the data is within some reasonable range and removes any
    that don't appear to be. Does this by computing the quantiles of the N_21 statistic.

    Uses frequency values for GPS_L1 and GPS_L2. Not usable with GLONASS currently.

    Parameters
    ----------
    station_matrix : numpy.ndarray
        The matrix pulled from scenario._station_data for a single station.

    Returns : bool
        True means quality check passed. False means to remove station.
    -------
    '''
    f1 = laika.constants.GPS_L1; f2 = laika.constants.GPS_L2;
    C = laika.constants.SPEED_OF_LIGHT; lam1 = C/f1; lam2 = C/f2;
    Fratio = (f1-f2)/(f1+f2)
    #
    Phi_21 = station_matrix['L1C'] - station_matrix['L2C']
    R1_t = station_matrix['C1C']
    R2_t = numpy.where(numpy.logical_not(numpy.isnan(station_matrix['C2C'])),
                       station_matrix['C2C'], station_matrix['C2P'])
    N_21_t = Phi_21 - Fratio * (R1_t / lam1 + R2_t / lam2)
    # nan_count = numpy.sum(numpy.isnan(N_21_t))
    quants =  numpy.quantile(N_21_t[numpy.where(numpy.logical_not(numpy.isnan(N_21_t)))], numpy.array([0., 0.05, 0.5, 0.95, 1. ]))
    return numpy.max(numpy.abs(quants)) <= max_absolute_n21_val

def populate_data_add_satellite_info(st_mat, cache_dir):
    '''
    This step does the equivalent of what the laika .process() command used to do. It has been hard to get this quite
    right but for the most part it works. If the satellite ephemeris data is unavailable then the is_processed value
    stays at 0. Does not return anything, but modified 'st_mat' in-place.

    :param st_mat: station_data matrix. I.e. the structured numpy array stored in self._station_data[...] for a
                    particular station. Should already be sorted by PRN, then tick, but we do it anyway just to
                    be sure.
    :param cache_dir: cache folder to look in for old ephemeris data and to store new ones.
    :return:
    '''
    thisdog = laika.astro_dog.AstroDog(cache_dir=cache_dir)
    st_mat.sort(axis=0, order=['prn','tick'])

    # First, make a list of GPStime objects to match the rows of st_mat:
    adj_sec = st_mat['recv_time_sec'] - st_mat['C1C'] / constants.SPEED_OF_LIGHT
    gps_times = list(map(lambda x: GPSTime(week=st_mat['recv_time_week'][x].item(), tow=adj_sec[x].item()),
                       range(st_mat.shape[0])))

    # Then pull the data and stick it in the matrix:
    for i in range(st_mat.shape[0]):
        si = thisdog.get_sat_info(st_mat['prn'][i].item(), gps_times[i]);
        if si is None:
            continue
        st_mat['sat_clock_err'][i] = si[2]
        st_mat['sat_pos_x'][i] = si[0][0]; st_mat['sat_pos_y'][i] = si[0][1]; st_mat['sat_pos_z'][i] = si[0][2]
        st_mat['sat_vel_x'][i] = si[1][0]; st_mat['sat_vel_y'][i] = si[1][1]; st_mat['sat_vel_z'][i] = si[1][2]
        st_mat['is_processed'][i] = 1;
    del thisdog

def get_vtec_data_dense(scenario, conn_map=None, biases=None):
    '''Same as the following function but for dense data structure...

        - This method could definitely be sped up using either some
            numpy functions or writing a C routine. Lots of math in
            python here.

    TODO: Move this function into the ScenarioInfoDense class'''
    t_start = datetime.now()
    # Make a helper function to calculate vtec for a station-prn for all ticks:
    def vtec_for(station, prn, conns=None, biases=None):
        '''Gets vTEC for a single station-PRN combo...'''
        if biases:
            station_bias = biases.get(station, 0)
            sat_bias = biases.get(prn, 0)
        else:
            station_bias, sat_bias = 0, 0
        tick_list = scenario.get_tick_list_for_prn(station, prn)

        for i in tick_list:
            np_meas = scenario.get_measure(station, prn, i)
            if conns:
                # case (1): good connection & good measurement
                if np_meas and conns[i] and ((conns[i].n1 and conns[i].n2) or conns[i].offset):
                    res = tec.calc_vtec( scenario, station, prn, i,
                        n1=conns[i].n1, n2=conns[i].n2, rcvr_bias=station_bias, sat_bias=sat_bias,
                        offset=conns[i].offset,
                    )  # --> returns (dat, loc, slant)
                    if res is None:
                        # case (2): Good connection but bad measurement for some reason
                        res_np = (math.nan, None, None, None, math.nan, math.nan, 0)
                    else:
                        res_np = (res[0], res[1][0], res[1][1], res[1][2], res[2], math.nan, 0)
                else:
                    res_np = (math.nan, None, None, None, math.nan, math.nan, 0)
            elif np_meas:
                # case (3): No connection but do have a measurement
                res = tec.calc_vtec( scenario, station, prn, i )
                if res is None:
                    # Computation failed for some reason
                    res_np = (math.nan, None, None, None, math.nan, math.nan, 0)
                else:
                    res_np = (res[0], res[1][0], res[1][1], res[1][2], res[2], math.nan, 0)
            else:
                res_np = (math.nan, None, None, None, math.nan, math.nan, 0)
            #
            # --> Now update the station_vtecs data in the scenario:
            data_row = scenario.get_measure(station, prn, i, row_only=True)
            scenario.station_vtecs[station][data_row] = res_np

    # *** Iterate over all the stations ***
    for station in scenario.stations:
        # print('\r' + ' '*40, end = '\r')
        print('Station: %s' % station)
        this_prn_list = scenario.get_prn_list_for_station(station)
        # *** Iterate over PRNs ***
        for prn in this_prn_list:
            print('...stn: %s (%s of %s), prn: %s  (%s of %s)...  Elapsed time: %.10s' % (
                station, scenario.stations.index(station), len(scenario.stations), prn, this_prn_list.index(prn),
                len(this_prn_list), datetime.now()-t_start), end = '\r')
            # Use the connection map if we have it, otherwise don't.
            if conn_map:
                if station not in conn_map:
                    break  # no connections... ignore this
                vtec_for(station, prn, conns=conn_map[station][prn], biases=biases)
            else:
                vtec_for(station, prn, biases=biases)
    print('')
    return

# def get_vtec_data(scenario, conn_map=None, biases=None):
#     '''
#     Iterates over (station, PRN) pairs and computes the VTEC for each one. VTEC here takes the form of a tuple of
#     lists in the form: (locs, dats, slants) where each one is a list of values of length max-tick-for-station-prn.
#     In other words, for every tick in the range of ticks seen for the (station, PRN) pair, we have a (loc, dat, slant)
#     triple, although they are each in their own vector.
#     Here 'loc' is the location of the ionosphere starting point. 'data' is the vtec calculation, and 'slatn' is the
#     slant_to_vertical conversion factor.
#     :param scenario:
#     :param conn_map:
#     :param biases:
#     :return:
#     '''
#     if scenario.station_data_structure=='dense':
#         # in dense form it works a bit differently, lines up with the structured nparray created initially.
#         get_vtec_data_dense(scenario, conn_map, biases)
#         return
#     station_vtecs = defaultdict(dict)   # The eventual output
#     def vtec_for(station, prn, conns=None, biases=None):
#         if biases:
#             station_bias = biases.get(station, 0)
#             sat_bias = biases.get(prn, 0)
#         else:
#             station_bias, sat_bias = 0, 0
#         # Only bother if the particular (station, prn) has not been done yet:
#         if prn not in station_vtecs[station]:
#             dats = []
#             locs = []
#             slants = []
#             if scenario.station_data[station].get(prn):
#                 end = max(scenario.station_data[station][prn].keys())
#             else:
#                 end = 0
#             for i in range(end):    # iterate over integers in the range of ticks
#                 measurement = scenario.station_data[station][prn][i]
#                 # if conns specified, require ambiguity data
#                 if conns:
#                     # if we have a good measurement that's part of a connection and has *a* measure of offset
#                     if measurement and conns[i] and ((conns[i].n1 and conns[i].n2) or conns[i].offset): # and numpy.std(conns[i].n1s) < 3:
#                         res = tec.calc_vtec(
#                             scenario,
#                             station, prn, i,
#                             n1=conns[i].n1,
#                             n2=conns[i].n2,
#                             rcvr_bias=station_bias,
#                             sat_bias=sat_bias,
#                             offset=conns[i].offset,
#                         ) # --> returns (dat, loc, slant)
#                         if res is None:
#                             locs.append(None)
#                             dats.append(math.nan)
#                             slants.append(math.nan)
#                         else:
#                             dats.append(res[0])
#                             locs.append(res[1])
#                             slants.append(res[2])
#                     else:
#                         locs.append(None)
#                         dats.append(math.nan)
#                         slants.append(math.nan)
#
#                 elif measurement: # measurement but not connection
#                     res = tec.calc_vtec(scenario, station, prn, i)
#                     if res is None:
#                         locs.append(None)
#                         dats.append(math.nan)
#                         slants.append(math.nan)
#                     else:
#                         dats.append(res[0])
#                         locs.append(res[1])
#                         slants.append(res[2])
#                 else:
#                     locs.append(None)
#                     dats.append(math.nan)
#                     slants.append(math.nan)
#
#             station_vtecs[station][prn] = (locs, dats, slants)
#         return station_vtecs[station][prn]
#
#     for station in scenario.stations:
#         print(station)
#         for prn in satellites:
#             # Use the connection map if we have it, otherwise don't.
#             if conn_map:
#                 if station not in conn_map:
#                     break  # no connections... ignore this
#                 vtec_for(station, prn, conns=conn_map[station][prn], biases=biases)
#             else:
#                 vtec_for(station, prn, biases=biases)
#     return station_vtecs

def correct_vtec_data(scenario, vtecs, sat_biases, station_biases):
    '''Runs through the station_vtecs object and makes the correction using the sat_biases/station_biases values.
    Returns an object of identical structure but containing the corrected value.'''

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