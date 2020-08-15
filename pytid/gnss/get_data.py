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
import pandas as pd
import os
import pickle
import requests
from scipy.signal import butter, filtfilt
import zipfile
import ftplib

from laika import constants
from laika.downloader import download_cors_station, download_file
from laika.gps_time import GPSTime
from laika.lib import coordinates
from laika.rinex_file import RINEXFile, DownloadError
from laika.dgps import get_station_position
import laika.raw_gnss as raw

from pytid.gnss import tec
from pytid.utils.configuration import missile_tid_rootfold

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

def data_for_station(dog, station_name, date):
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
            rinex_obs_file = download_cors_station(time, station_name, cache_dir=dog.cache_dir)
        except (KeyError, DownloadError):
            pass

        if not rinex_obs_file:
            # station position not in CORS map, try another thing
            if station_name in extra_station_info:
                # station_pos = numpy.array(extra_station_info[station_name])
                rinex_obs_file = download_misc_igs_station(time, station_name, cache_dir=dog.cache_dir)
            else:
                raise DownloadError
    else:
        # station_pos = numpy.array(extra_station_info[station_name])
        rinex_obs_file = handlers[network](time, station_name, cache_dir=dog.cache_dir)
        
    obs_data = RINEXFile(rinex_obs_file, rate=30)
    # return station_pos, raw.read_rinex_obs(obs_data)
    return raw.read_rinex_obs(obs_data)

def meas_to_tuple(ms, stn, tick):
    return (stn, ms.prn, tick,                                                                              #indexes
            ms.observables.get('C1C', math.nan), ms.observables.get('C2C', math.nan),
            ms.observables.get('C2P', math.nan), ms.observables.get('C5C', math.nan),
            ms.observables.get('L1C', math.nan), ms.observables.get('L2C', math.nan),
            ms.observables.get('L5C', math.nan),
            ms.recv_time_sec, ms.recv_time_week,                                                            #recv_time
            ms.sat_clock_err,                                                                               #sat_clock_err
            ms.sat_pos[0].item(), ms.sat_pos[1].item(), ms.sat_pos[2].item(),                               #sat_pos
            ms.sat_vel[0].item(), ms.sat_vel[1].item(), ms.sat_vel[2].item()                                #sat_vel
            )


def data_for_station_npstruct(dog, station_name, date, prn_list=None, t0_date=None):
    '''
    This function will wrap the one above, but instead of returning this cumbersome list-of-lists, it
    will convert it into a structured numpy array.
    :param dog:             (self-explanatory)
    :param station_name:    (self-explanatory)
    :param date:            (self-explanatory, start date for this particular data pull)
    :param prn_list:        a list of specific prns that should be restricted to (e.g. GPS)
    :param t0_date:         (datetime) representing the tick-0 point. For calibrating ticks.

    :return: a structured numpy array containing the results.
    '''
    t0 = GPSTime.from_datetime(date) if t0_date is None else GPSTime.from_datetime(t0_date)
    def meas_tick_calc(ms):
        return round((ms.recv_time - t0)/30.0)

    raw_obs = data_for_station(dog, station_name, date)

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
                          ('sat_pos_z', 'f8'), ('sat_vel_x', 'f8'), ('sat_vel_y', 'f8'), ('sat_vel_z', 'f8')])
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

    return dfs.reshape((tot_meas_ct,1))

def get_dates_in_range(start_dt, durr):
    '''returns a list of datetime objects separated by a day that fall in the range of start_dat and durration.'''
    dates = []
    date_var = start_dt
    while date_var < start_dt + durr:
        dates.append(date_var)
        date_var += timedelta(days=1)
    return dates

def get_cors_station_lists_for_day(dt):
    '''For a particular day, pulls a list of CORS stations that have data posted for each of the two CORS sites.
    Technically it just returns a directory listing limited only to items of length 4, but this will do.'''
    # f1 = 'ftp://geodesy.noaa.gov/cors/rinex/'
    # f2 = 'ftp://alt.ngs.noaa.gov/cors/rinex/'
    print('getting cors station list for %s' % dt)
    f1='geodesy.noaa.gov'
    f2='alt.ngs.noaa.gov'
    day_folder = "/cors/rinex/" +  dt.strftime('%Y/%j/')
    ftp = ftplib.FTP(f1, "anonymous", "")
    f1nlst = ftp.nlst(day_folder)
    ftp.quit()
    ftp = ftplib.FTP(f2, "anonymous", "")
    f2nlst = ftp.nlst(day_folder)
    ftp.quit()
    f1sta = list(filter(lambda x: len(x)==4, list(map(lambda x: x.replace(day_folder,''), f1nlst))))
    f2sta = list(filter(lambda x: len(x) == 4, list(map(lambda x: x.replace(day_folder, ''), f2nlst))))
    return f1sta, f2sta

def cors_download_commands(stns, dt_list, out_script='bash_cors_obs_download.sh', cache_dir='~/.gnss_cache'):
    '''
    :param stns:
    :param dt_list:
    :param out_script: name of the file for the bash script. It will natrually be thrown in the 'scripts' folder.
    :param cache_dir: the one from AstroDog
    :return:
    '''
    def get_dt_cors_stns(dt):
        '''Helper function to ID the subset of <stns> that will have CORS files to download.'''
        cors_stns = get_cors_station_lists_for_day(dt)
        my_cors_stns_tup = (set(stns).intersection(list(cors_stns[0])), set(stns).intersection(list(cors_stns[1])))
        my_cors_stns = list(set(my_cors_stns_tup[0]).union(set(my_cors_stns_tup[1])))
        return my_cors_stns

    url_base1='ftp://geodesy.noaa.gov'
    url_base2='ftp://alt.ngs.noaa.gov'
    bash_script = open(os.path.join(missile_tid_rootfold, 'scripts', out_script), 'w')
    if cache_dir[0]=='~':
        cache_dir = os.path.expanduser(cache_dir)

    def cors_url_to_bash(stn, tgt_date):
        filename = stn + tgt_date.strftime("%j0.%yo.gz")
        cors_file_url1 = url_base1 + "/cors/rinex/" + tgt_date.strftime('%Y/%j/') + stn + '/' + filename
        cors_file_url2 = url_base2 + "/cors/rinex/" + tgt_date.strftime('%Y/%j/') + stn + '/' + filename
        dl_target_fold = os.path.join(cache_dir, 'cors_obs', tgt_date.strftime('%Y'), tgt_date.strftime('%j'), stn)
        l1ct = bash_script.write('wget %s -P %s -nc \n' % (cors_file_url1, dl_target_fold))
        l2ct = bash_script.write('if [ $? -eq 0 ]; then gunzip -c %s > %s ; \n' %
                                 (os.path.join(dl_target_fold, filename), os.path.join(dl_target_fold, filename)[:-3]))
        l3ct = bash_script.write('else wget %s -P %s -nc ; \n' % (cors_file_url2 , dl_target_fold))
        l4ct = bash_script.write('if [ $? -eq 0 ]; then gunzip -c %s > %s   ; fi; fi;\n\n' %
                                 (os.path.join(dl_target_fold, filename), os.path.join(dl_target_fold, filename)[:-3]))

    for my_dt in dt_list:
        date_cors_stns = get_dt_cors_stns(my_dt)
        print('%s - ' % my_dt, end='\r', flush=True)
        for s in date_cors_stns:
            print('%s - %s' % (my_dt,s), end='\r', flush=True)
            cors_url_to_bash(s, my_dt)
    bash_script.close()

def get_igs_station_lists_for_day(dt):
    f1=('garner.ucsd.edu' , '/archive/garner/rinex/')
    f2=('data-out.unavco.org','/pub/rinex/obs/')
    f3=('nfs.kasi.re.kr','/gps/data/daily/')
    f4=('igs.gnsswhu.cn','/pub/gps/data/daily/')
    f5=('cddis.nasa.gov','/gnss/data/daily/')

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
        self.start_date = start_date
        self.duration = duration
        self.stations = stations
        self._station_locs = None
        self._station_data = None
        self._clock_biases = None
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
            self.populate_data(method=self.station_data_structure)
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
        for stn in self.stations:
            cache_name = "cached/%s_%s_%s_to_%s" % ( cache_file_prefix, stn,
                self.start_date.strftime("%Y-%m-%d"), (self.start_date + self.duration).strftime("%Y-%m-%d"))
            if os.path.exists(cache_name):
                self.station_data_sources[stn]=('cached_pickle', cache_name)
                pickle_file_ct += 1

        # Then pull down the CORS lists for each day:
        cors_lists = dict.fromkeys(range(len(self.date_list)))
        for i in range(len(self.date_list)):
            cors_lists_d = get_cors_station_lists_for_day(self.date_list[i])
            cors_lists[i] = cors_lists_d
        cors_ftp_ct = 0
        for stn in self.stations:
            if self.station_data_sources[stn] is None:
                stn_daily_options = []
                stn_cors_ftp_ok_alldays = True
                for i in range(len(self.date_list)):
                    in_f1 = stn in cors_lists[i][0]; in_f2 = stn in cors_lists[i][1]
                    if (not in_f1) and (not in_f2):
                        stn_cors_ftp_ok_alldays = False
                    stn_daily_options.append((in_f1, in_f2))
                if stn_cors_ftp_ok_alldays:
                    self.station_data_sources[stn] = ('cors_ftp', stn_daily_options)
                    cors_ftp_ct += 1

        print('%s total stns, %s are pickled, %s are on CORS ftp sites.' % (len(self.station_data_sources),
                                                                            pickle_file_ct, cors_ftp_ct))

    def populate_data(self, method='dict'):
        '''
        wrapper for two methods to populate the data now.
        :param method: Must be one of ('dict', 'dense')
        :return:
        '''
        self._station_locs = {}
        self._station_data = {}


        self.bad_stations = []
        self.populate_station_locs()

        #Choose which populate subroutine to run.
        if method=='dict':
            self.populate_data_dict()
        elif method=='dense':
            self.populate_data_dense()

        for bad_station in self.bad_stations:
            self.stations.remove(bad_station)

    def populate_data_dict(self):
        self._station_data = {}
        for station in self.stations:
            print(station)
            cache_name = "cached/stationdat_%s_%s_to_%s" % (
                station,
                self.start_date.strftime("%Y-%m-%d"),
                (self.start_date + self.duration).strftime("%Y-%m-%d")
            )
            # Check to see if we have cached this thing. If so, recover it from there
            if os.path.exists(cache_name):
                self._station_data[station] = pickle.load(open(cache_name, "rb"))
                # try:
                #     self.station_locs[station] = get_station_position(station, cache_dir=self.dog.cache_dir)
                # except KeyError:
                #     self.station_locs[station] = numpy.array(extra_station_info[station])
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
                                                offset=int((date - self.start_date).total_seconds()/30)
                                            )
                    # self.station_locs[station] = loc
                except (ValueError, DownloadError):
                    print("*** error with station " + station)
                    self.bad_stations.append(station)
                date += timedelta(days=1)
            os.makedirs("cached", exist_ok=True)
            pickle.dump(self._station_data[station], open(cache_name, "wb"))

    def populate_data_dense(self):
        '''
        In this version, the outermost object is a pandas data-frame instead of being a nested-dict.
        :return:
        '''
        self._station_data = {}
        self.prepare_data_sources('stationdat_dense')

        cors_stns = []; pickle_stns=[]
        for cs in self.stations:
            if self.station_data_sources[cs] is None:
                continue
            elif self.station_data_sources[cs][0]=='cors_ftp':
                cors_stns.append(cs)
            elif self.station_data_sources[cs][0]=='cached_pickle':
                pickle_stns.append(cs)

        for station in pickle_stns:
            print(station)
            if self.station_data_sources[station][0]=='cached_pickle':
                cache_name=self.station_data_sources[station][1]
                if os.path.exists(cache_name):
                    self._station_data[station] = pickle.load(open(cache_name, "rb"))

            # Start populating this dictionary.
        for station in cors_stns:
            date = self.start_date
            data_by_day = []
            while date < self.start_date + self.duration:
                print('%s\t%s' % (station, date.strftime("%Y-%m-%d")), end='\r')
                try:
                    dfs=data_for_station_npstruct(self.dog, station, date, self.prn_list, self.start_date)
                    data_by_day.append(dfs)
                except (ValueError, DownloadError):
                    print("*** error with station " + station)
                    self.bad_stations.append(station)
                date += timedelta(days=1)

            if station not in self.bad_stations:
                # Combine the data for each of the three days.
                self._station_data[station]=numpy.vstack(tuple(data_by_day))
                # save this thing to the cached folder:
                os.makedirs("cached", exist_ok=True)
                pickle.dump(self._station_data[station], open(cache_name, "wb"))



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