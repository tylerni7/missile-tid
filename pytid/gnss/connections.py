"""
each satellite <-> station "connection" has fixed values for
integer ambiguities. Break this out into an easy-to-use class
"""
from collections import defaultdict
from laika import helpers
from laika.lib import coordinates
import math, datetime
import numpy
import ruptures as rpt

from pytid.gnss import ambiguity_correct
from pytid.gnss import get_data
from pytid.gnss import tec

CYCLE_SLIP_CUTOFF_MW = 5
CYCLE_SLIP_CUTOFF_DELAY = 1.0
CYCLE_SLIP_MW_MUSIGMA_FACTOR = 5.0
MIN_CON_LENGTH = 20  # 10 minutes worth of connection
DISCON_TIME = 4      # cycle slip for >= 5 samples of no info
FULL_CORRECTION = False  # whether to do the full laika "correct" method, largely unnecessary
EL_CUTOFF = 0.30


def contains_needed_info(measurement):
    '''Make sure that all four observables are valid numerical vals. Operates on the old nested-dict station_data
    structure.'''
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    needed = {'L1C', 'L2C', 'C1C', chan2}
    for need in needed:
        if math.isnan(observable.get(need, math.nan)):
            return False
    return True

def contains_needed_info_dense(np_meas):
    '''
    Make sure that all four observables are valid numerical vals.

    :param np_meas: single row of the station_data np structured array.
    :return:
    '''
    chan2 = 'C2C' if not numpy.isnan(np_meas['C2C'][0]) else 'C2P'
    needed = {'L1C', 'L2C', 'C1C', chan2}
    for need in needed:
        if numpy.isnan(np_meas[need][0]):
            return False
    return True

def is_processed_dense(np_meas):
    '''Checks to see if any of the last 7 fields are not NaN'''
    sat_fields = ['sat_clock_err', 'sat_pos_x', 'sat_pos_y', 'sat_pos_z', 'sat_vel_x', 'sat_vel_y', 'sat_vel_z']
    return not numpy.all(list(map(lambda y: numpy.isnan(np_meas[y])[0], sat_fields)))

class Connection:
    def __init__(self, scenario, station, prn, tick0, tickn, filter_ticks=True):
        self.scenario = scenario
        if scenario:
            self.loc = scenario.station_locs[station]
        self.station = station
        self.prn = prn
        self.tick0 = tick0 #at init, first possible tick
        self.tickn = tickn #at init, largest tick that exists
        self.ticks = None
        self.break_reason = None

        # integer ambiguities... what we really want
        self.n1 = None
        self.n2 = None

        # "raw" offset: the difference from the code phase tec values
        self.offset = None
        self.werrs = []

        # TODO: this could incorporate phase windup as well?
        if filter_ticks and scenario:
            if self.scenario.station_data_structure=='dense':
                prn_rows = numpy.where(scenario.station_data[station]['prn']==prn)[0]
                self.filter_ticks_dense(scenario.station_data[station][prn_rows])
            else:
                self.filter_ticks(scenario.station_data[station][prn])

    def filter_ticks(self, this_data):
        '''
        NOTE: THIS FUNCTION CURRENTLY ONLY APPLIES TO THE OLD DATA STRUCTURE. There is some logic in here to use the
        new dense structure, but it has been replaced by a whole new algorithm in the <>_dense function below. That
        makes the logic in this function essentially deprecated (MN, 2/24/21).

        Starts with an assumed 'connection' of length zero at the first time point ('tick'). Runs through the list
        of ticks (integers) in increasing order. Skips ticks where the satelite elevation was too low. With each tick,
        judges whether the connection has 'dropped' either due to a) DISCONNECTION, or b) CYCLE_SLIP. If it has dropped,
        it stops counting and the connection is considered 'complete' (i.e. initialized).

        The parameter `this_data` can be formatted in one of two ways depending on the data_structure used by the
        scenario. If the data_structure is 'dense' (i.e., the structured numpy array), then `this_data` should be the
        sub-array restricted to the PRN-Station combo given at initialization. Otherwise it is the dictionary keyed by
        ticks.
        '''
        if self.ticks is not None:
            return

        self.ticks = []
        last_seen = self.tick0
        last_mw = None
        last_tec = None

        tick_to_row = {}
        if self.scenario.station_data_structure=='dense':
            tick_to_row = dict(zip(this_data['tick'][:,0], numpy.arange(this_data.shape[0])))

        for tick in range(self.tick0, self.tickn):
            # First check that the tick a) exists, b) has all 4 observables, and c) has the satelite info
            #   --> have to do this differently depending on the data structure
            if self.scenario.station_data_structure=='dense':
                if tick not in tick_to_row:
                    continue
                tick_r = tick_to_row[tick]
                if not contains_needed_info_dense(this_data[tick_r]):
                    continue
                if not is_processed_dense(this_data[tick_r]):
                    continue
                tick_sat_pos = list(map(lambda y: this_data[tick_r][y][0], ['sat_pos_x','sat_pos_y','sat_pos_z']))
                el = self.scenario.station_el(self.station, tick_sat_pos)
            else:
                if not this_data[tick]:
                    continue
                if not contains_needed_info(this_data[tick]):
                    continue
                if not this_data[tick].processed:
                    if not this_data[tick].process(self.scenario.dog):
                        continue
                # ignore "bad connections" (low elevation)
                if FULL_CORRECTION and not this_data[tick].corrected:
                    this_data[tick].correct(self.loc, self.scenario.dog)
                el = self.scenario.station_el(self.station, this_data[tick].sat_pos)

            if el < EL_CUTOFF:
                continue

            # detect slips due to long disconnection time (5 ticks)
            if tick - last_seen > DISCON_TIME:
                self.break_reason = 'discon'
                break
            last_seen = tick

            # detect cycle slips due to changing N_w (Cutoff=10 by default)
            if self.scenario.station_data_structure=='dense':
                mw, _ = tec.melbourne_wubbena_dense(self.scenario.dog, this_data[tick_r])
            else:
                mw, _ = tec.melbourne_wubbena(self.scenario.dog, this_data[tick])
            if last_mw is not None and abs(last_mw - mw) > CYCLE_SLIP_CUTOFF_MW:
                print("cycle slip: {0}-{1}@{2} mw jump of {3:0.2f}".format(
                    self.station, self.prn, tick, last_mw - mw
                ))
                self.break_reason = 'mw'
                break
            last_mw = mw

            # OR cycle slips from big changes in TEC (diff of 0.3m. This is currently causing mmany of the slips detected)
            if self.scenario.station_data_structure == 'dense':
                delay, _ = tec.calc_carrier_delay_dense(self.scenario.dog, this_data[tick_r])
            else:
                delay, _ = tec.calc_carrier_delay(self.scenario.dog, this_data[tick])
            if last_tec is not None and abs(last_tec - delay) > CYCLE_SLIP_CUTOFF_DELAY:
                print("cycle slip: {0}-{1}@{2} delay jump of {3:0.2f}".format(
                    self.station, self.prn, tick, last_tec - delay
                ))
                self.break_reason = 'tec_jump'
                break
            last_tec = delay

            # all good
            self.ticks.append(tick)

        # Update the local property holding the first and last tick:
        if self.ticks:
            self.tick0 = self.ticks[0]
            self.tickn = self.ticks[-1]
        else:
            self.ticks = [tick]
            self.tick0 = tick
            self.tickn = tick

    def filter_ticks_dense(self, this_data):
        '''
        Starts with an assumed 'connection' of length zero at the first time point ('tick'). Runs through the list
        of ticks (integers) in increasing order. Skips ticks where the satelite elevation was too low. With each tick,
        judges whether the connection has 'dropped' either due to a) DISCONNECTION, or b) CYCLE_SLIP. If it has dropped,
        it stops counting and the connection is considered 'complete' (i.e. initialized).

        The cycle-slip detector here comes from the mean-variance Melbourne-Wubbena algorithm on Navipedia, at:
            https://gssc.esa.int/navipedia/index.php/Detector_based_in_code_and_carrier_phase_data:_The_Melbourne-W%C3%BCbbena_combination

        The parameter `this_data` is assumed to be dense (i.e., the structured numpy array).
        '''
        if self.ticks is not None:
            return

        self.ticks = []
        last_seen = self.tick0; last_mw = None; last_tec = None; mw_mean=0.; mw_sigma2=None; mw_ct=0;

        tick_to_row = dict(zip(this_data['tick'][:,0], numpy.arange(this_data.shape[0])))

        for tick in range(self.tick0, self.tickn):
            # First check that the tick a) exists, b) has all 4 observables, and c) has the satelite info
            #   --> have to do this differently depending on the data structure
            print('', end = '\r')
            print('%s - %s - tick %s (%s to %s): ' % (self.station, self.prn, tick, self.tick0, self.tickn), end ='')
            if tick not in tick_to_row: #tick doesn't exist
                continue
            tick_r = tick_to_row[tick]
            if not contains_needed_info_dense(this_data[tick_r]): #doesn't have all the data needed
                continue
            if not is_processed_dense(this_data[tick_r]): #must have sat_info
                continue
            tick_sat_pos = list(map(lambda y: this_data[tick_r][y][0], ['sat_pos_x','sat_pos_y','sat_pos_z']))
            el = self.scenario.station_el(self.station, tick_sat_pos)

            if el < EL_CUTOFF:
                continue

            # detect slips due to long disconnection time (5 ticks)
            if tick - last_seen > DISCON_TIME:
                self.break_reason = 'discon'
                break
            last_seen = tick

            # detect cycle slips due to changing N_w (Cutoff=10 by default)
            #   ...algorithm from: https://gssc.esa.int/navipedia/index.php/Detector_based_in_code_and_carrier_phase_data:_The_Melbourne-W%C3%BCbbena_combination
            #   a) Get current tick MW and lambda_W values, calc MW variance:
            mw, lambda_W = tec.melbourne_wubbena_dense(self.scenario.dog, this_data[tick_r])
            mw_ct += 1
            if mw_sigma2 is None:
                mw_sigma2 = (0.5*lambda_W)**2

            #   b) Update cumulative mean/variance values:
            mw_sigma2 = (mw_ct - 1.0) / mw_ct * mw_sigma2 + 1.0 / mw_ct * (mw - mw_mean)**2
            mw_mean = (mw_ct-1.0)/mw_ct * mw_mean + 1.0/mw_ct * mw

            #   c) If the current value exceeds the mean by 5.0 * sigma, then terminate:
            if mw_ct>6 and abs(mw - mw_mean) > CYCLE_SLIP_MW_MUSIGMA_FACTOR * (mw_sigma2**0.5):
                print("cycle slip: {0}-{1}@{2} mw mu/sigma ratio of of {3:0.2f}".format(
                    self.station, self.prn, tick, (mw - mw_mean) / (mw_sigma2**0.5)
                ), end = '\r')
                self.break_reason = 'mw_test'
                break

            # *** Previous MW Logic, for reference ***
            # if last_mw is not None and abs(last_mw - mw) > CYCLE_SLIP_CUTOFF_MW:
            #     print("cycle slip: {0}-{1}@{2} mw jump of {3:0.2f}".format(
            #         self.station, self.prn, tick, last_mw - mw
            #     ))
            #     self.break_reason = 'mw'
            #     break
            last_mw = mw

            # OR cycle slips from big changes in TEC (diff of 0.3m. This is currently causing mmany of the slips detected)
            # *** Note: changed this to 1.0m on 10/15/2020 - MN ***
            delay, _ = tec.calc_carrier_delay_dense(self.scenario.dog, this_data[tick_r])

            if last_tec is not None and abs(last_tec - delay) > CYCLE_SLIP_CUTOFF_DELAY:
                print("cycle slip: {0}-{1}@{2} delay jump of {3:0.2f}".format(
                    self.station, self.prn, tick, last_tec - delay
                ), end = '\r')
                self.break_reason = 'tec_jump'
                break
            last_tec = delay

            # all good
            self.ticks.append(tick)

        # Update the local property holding the first and last tick:
        if self.ticks:
            self.tick0 = self.ticks[0]
            self.tickn = self.ticks[-1]
        else:
            self.ticks = [tick]
            self.tick0 = tick
            self.tickn = tick

    def set_ticks(self, ticks_filtered):
        '''
        Method to set the ticks directly rather than running the filtering method every time.
        This is mostly so I can load connection objects from saved data without re-running the
        filtering routine every time.
        :param ticks_filtered:
        :return:
        '''
        self.ticks = ticks_filtered
        self.tick0 = self.ticks[0]
        self.tickn = self.ticks[-1]


def test_conn_for_changepoints(cn, scenario, count_bps_only=False):
    '''
    Runs the basic ruptures changepoint test on a given scenario and connection using the vector of Melbourne-Wubenna
    values. Returns either the list of change-points or the count of them (if 'count_bps_only' is True).

    The rpt.Binseg command will always return at least a list with the last element in the sequence only. The first
    element (0-indexed) is never included, so we add it.

    Parameters
    ----------
    cn : Connection
        The connection to check for M-W changepoints
    scenario : ScenarioInfoDense
        The associated scenario object
    count_bps_only : bool
        If True, return the number of change-points rather than the list o them.

    Returns : int / list (as specified)
    -------

    '''
    mw = tec.melbourne_wubbena_vector_from_conn(scenario, cn)
    algo = rpt.Binseg(model='l2').fit(mw)
    bkps = [0,]+ algo.predict(pen=mw.shape[0]/numpy.log(mw.shape[0]))
    if count_bps_only:
        return len(bkps)-2
    else:
        return bkps

def find_and_remove_remaining_cycle_slips(scenario):
    '''
    Runs the ruptures change-point test on every connection object. For any with change-points, splits
    those connections into separate individual ones.

    Parameters
    ----------
    scenario

    Returns
    -------

    '''
    start_time = datetime.datetime.now()
    # --- 1) Get a count of change-points for every individual connection:
    print('1) Getting change-point counts for all connections.')
    conn_chgpt_counts = [test_conn_for_changepoints(c, scenario, True) for c in scenario.conns]
    # --- 2) Get a list of indices of the connections needing edit:
    chgpt_conn_inds = list(numpy.where(numpy.array(conn_chgpt_counts) > 0)[0])
    chgpt_conn_inds.sort(reverse=True)
    # --- 3) Make a dictionary of actual change-point locations in the connection:
    print('3) Making dict of change-point locations.')
    conn_chgpt_loc_data = {k: test_conn_for_changepoints(scenario.conns[k], scenario) for k in chgpt_conn_inds}

    # --- 4) Now go through each connection in need of edit, remove it from scenario, then split it
    #        and add it to a new list of connections. At the end, add these to scneario.conns:
    print('4) Fixing connections with uncaught cycle-slips.')
    new_connections = []
    for i in range(len(chgpt_conn_inds)):
        print('  ...connection %d of %d fixed.' % (i, len(chgpt_conn_inds)), end = '\r')
        # ***** THIS STEP IS IMPORTANT *****
        #   --> The old connection object must be removed from the list via either .pop() or remove()
        this_conn = scenario.conns.pop(chgpt_conn_inds[i])
        new_conns_tmp = split_connection(this_conn, conn_chgpt_loc_data[chgpt_conn_inds[i]])
        new_connections += new_conns_tmp
        del new_conns_tmp

    print(' '*30)
    print('DONE!')
    # --- 5) Finally, add the list of new connections to scenario.conns:
    scenario.conns += new_connections

class Group:
    """
    Two stations and two satellites such that all station X prn pairs have a connection
    """
    def __init__(self, station_data, connections):
        self.station_data = station_data
        self.connections = sorted(connections, key=lambda x:(x.station, x.prn))
        self.station1 = self.connections[0].station
        self.station2 = self.connections[2].station
        self.prn1 = self.connections[0].prn
        self.prn2 = self.connections[1].prn
        self.ticks = list(set.intersection(*[set(c.ticks) for c in connections]))

    def double_differences(self, calculator):
        return [
            ambiguity_correct.double_difference(
                calculator,
                self.station_data,
                self.station1,
                self.station2,
                self.prn1,
                self.prn2,
                tick
            ) for tick in self.ticks
        ]

    def double_difference(self, calculator):
        return numpy.mean([
            ambiguity_correct.double_difference(
                calculator,
                self.station_data,
                self.station1,
                self.station2,
                self.prn1,
                self.prn2,
                tick
            ) for tick in self.ticks
        ])

    @property
    def ddn1(self):
        return (
            self.connections[0].n1
            - self.connections[1].n1
            - self.connections[2].n1
            + self.connections[3].n1
        )

    @property
    def ddn2(self):
        return (
            self.connections[0].n2
            - self.connections[1].n2
            - self.connections[2].n2
            + self.connections[3].n2
        )

    @property
    def ddwide(self):
        return self.ddn1 - self.ddn2

    @property
    def ddnarrow(self):
        return self.ddn1 + self.ddn2

    def __repr__(self):
        return (
            f"< {self.station1}-{self.station2} "
            f"{self.prn1}-{self.prn2} "
            f"[{self.ticks[0]}-{self.ticks[-1]}] >"
        )


def split_connection(conn, tick_positions, verbose=True):
    '''
    Takes a connection object and a list of tick positions and splits the connection into new
    connections at each of the break-points listed in 'tick_positions'.

    The connection object passed in here must be popped off from whatever list or dict
    it may have been a part of. It will be deleted before this routine is over.

    Parameters
    ----------
    conn : Connection
        The large connection to be broken up.
    tick_positions : list[<int>]
        List of points to split the big connection.
    verbose : bool
        Controls reporting level (default = True)

    Returns : list of Connection objects
    -------

    '''
    n_orig_ticks = len(conn.ticks)
    orig_tick_vector = conn.ticks.copy()
    new_conn_set = []
    tick_breaks = [i for i in tick_positions if i>0 and i<n_orig_ticks]
    tick_breaks.sort()

    lower_bound_tick = 0
    # We will move up the list, with each new segment defining a new connection. The very
    #   last segment remaining will be the modified original connection.
    for i in range(len(tick_breaks)):
        putative_upper = tick_breaks[i] #upper bound *does* go inside the interval we are creating.
        # if verbose:
        #     print ('lower=%s, put_upper=%s' % (lower_bound_tick, putative_upper))
        if putative_upper - lower_bound_tick < MIN_CON_LENGTH:
            #too small, eliminating that interval
            lower_bound_tick = putative_upper #(here we an hang on to the last upper as the new lower because
                # it didn't get used for anything)
            continue
        newconn = Connection(None, conn.station, conn.prn, tick0=None, tickn=None, filter_ticks=False)
        newconn.ticks = orig_tick_vector[lower_bound_tick:(putative_upper+1)]
        newconn.tick0 = min(newconn.ticks)
        newconn.tickn = max(newconn.ticks)
        newconn.break_reason = 'melbourne_wubbena_beakpoint'
        newconn.loc = conn.loc.copy()
        new_conn_set.append(newconn)
        lower_bound_tick = putative_upper + 1

    remaning_ticks = orig_tick_vector[lower_bound_tick:]
    if len(remaning_ticks) >= MIN_CON_LENGTH:
        lastconn = Connection(None, conn.station, conn.prn, tick0 = None, tickn = None, filter_ticks=False)
        lastconn.ticks = remaning_ticks
        lastconn.tick0 = min(lastconn.ticks)
        lastconn.tickn = max(lastconn.ticks)
        lastconn.break_reason = conn.break_reason
        lastconn.loc = conn.loc.copy()
        new_conn_set.append(lastconn)

    del conn
    return new_conn_set

def make_connections_dense(scenario, station, prn, tick0, tickn):
    '''A separate version of the make_connections function for the dense structure.'''
    rows = numpy.where(scenario.station_data[station]['prn']==prn)[0]
    ticks = scenario.station_data[station]['tick'][rows][:,0]
    ticks_to_rows = dict(zip(ticks,rows))
    def next_valid_tick(ticki):
        return min([i for i in ticks_to_rows.keys() if i>ticki])
        # return scenario.station_data[station]['tick'][ticks_to_rows[ticki]+1][0]

    ticki = ticks[0]; tickn = ticks[-1];
    connections = []
    while ticki < tickn:
        con = Connection(scenario, station, prn, ticki, tickn)
        if not con.ticks:
            ticki = next_valid_tick(ticki)
            continue
        elif len(con.ticks) > MIN_CON_LENGTH:
            con.scenario = None #kill this reference so we can save these and not recompute them every time.
            connections.append(con)

        ticki = next_valid_tick(con.tickn)
    return connections

def make_connections(scenario, station, prn, tick0, tickn):
    """
    given data and a bunch of ticks, split it up into connections
    """
    if scenario.station_data_structure=='dense':
        return make_connections_dense(scenario, station, prn, tick0, tickn)

    def next_valid_tick(tick):
        '''Gets the next integer above `tick` in the dict station_data[station][prn]'''
        for i in range(tick + 1, tickn):
            if scenario.station_data[station][prn][i]:
                return i
        return tickn

    connections = []
    ticki = tick0
    # 1) make a connection from the ticks. If long enough, add to inventory. 2) chop off the ticks consumed. 3) If any
    #   ticks are left, goto (1).
    while ticki < tickn:
        con = Connection(scenario, station, prn, ticki, tickn)
        if not con.ticks:
            ticki = next_valid_tick(ticki)
            continue
        elif len(con.ticks) > MIN_CON_LENGTH:
            connections.append(con)
        ticki = next_valid_tick(con.tickn)
    return connections

def get_connections(scenario, skip=None, station_subset=None):
    '''
    For each (station,satelite), grabs the ticks and GNSS measurements (the innermost dict of station_data) and
    runs it through "make_connections()" to get a list of Connection objects.
    :param scenario: ScenarioInfo object
    :param skip: a list of stations that should be skipped.
    :return:
    '''
    def get_prns(stn):
        '''Returns a list of the PRNs in the scenario for a given station. If the data strcuture is `dense`
        then do it using numpy.unique, otherwise the old way.'''
        if scenario.station_data_structure=='dense':
            return numpy.unique(scenario.station_data[stn]['prn'])
        else:
            return scenario.station_data[station].keys()

    def get_tick_list(stn, prn):
        '''Returns an iterator over the ticks for which the station-prn combo is active. Sorted list.'''
        if scenario.station_data_structure=='dense':
            tks=scenario.station_data[stn]['tick'][numpy.where(scenario.station_data[stn]['prn']==prn)[0]][:,0]
            return tks
        else:
            return scenario.station_data[station][prn].keys()

    connections = []
    if station_subset is None:
        station_subset = scenario.stations
    for station in station_subset:
        print('\r --station %s (%s of %s): ' % (station, station_subset.index(station), len(station_subset)))
        if skip and station in skip:
            continue
        # for prn in scenario.station_data[station].keys():
        for prn in get_prns(station):
            # ticks = scenario.station_data[station][prn].keys()
            ticks = get_tick_list(station, prn)
            if len(ticks)==0:
                continue
            # try:
            connections += make_connections(
                scenario,
                station,
                prn,
                min(ticks),
                max(ticks)
            )
            # except:
            #     print('stn=%s, prn=%s, len(ticks)=%s, tick0=%s, tickn=%s' % (station, prn, len(ticks), min(ticks), max(ticks)))
    return connections

def get_groups(station_data, connections):
    """
    return lists of connection groups with 4 connections from
    2 stations and 2 satellites, and overlapping times
    """
    groups = []

    min_tick = min([con.tick0 for con in connections])
    max_tick = max([con.tickn for con in connections])

    def connections_including(tick):
        res = []
        for con in connections:
            if con.tick0 <= tick <= con.tickn:
                if tick in con.ticks:
                    res.append(con)
        return res

    def connections_to(candidates, prn):
        res = set()
        for con in candidates:
            if con.prn == prn:
                res.add(con)
        return res

    def partners_for(candidates, con):
        sta1 = con.station
        prn1 = con.prn

        same_station = set(con for con in candidates if con.station == sta1)

        for con2 in same_station:
            if con2 == con:
                continue

            prn2 = con2.prn
            # okay now we have station with two satellites
            # find one more station with the same two...
            eligible = connections_to(candidates - same_station, prn1)

            for con3 in connections_to(eligible, prn1):
                sta2 = con3.station
                lasts = [con for con in candidates if con.station == sta2 and con.prn == prn2]
                if lasts:
                    return {con2, con3, lasts[0]}
        return None

    # try to get everything paired up
    unpaired = set(connections)

    for tick in range(min_tick, max_tick):
        if tick % 100 == 0:
            print(tick, max_tick)
        candidates = set(connections_including(tick))
        need_pairing = candidates & unpaired

        for con in need_pairing:
            if con not in unpaired:
                # could have updated after we started iterating
                continue

            partners = partners_for(candidates, con)
            if not partners:
                # alone this tick
                continue
            groups.append( Group(station_data, partners | set([con])) )
            unpaired -= partners
            unpaired.remove(con)

    return groups, unpaired

diffs = []

# def correct_group(scenario, group):
#     '''
#     Applies the ambiguity correction calculated from a 'group' of station/prns.
#     :param scenario:
#     :param group:
#     :return:
#     '''
#     members = sorted(group, key=lambda x:(x.station, x.prn))
#     sta1 = members[0].station
#     sta2 = members[2].station
#     prn1 = members[0].prn
#     prn2 = members[1].prn
#
#     ticks = [members[i].ticks for i in range(4)]    # list of lists
#     res = ambiguity_correct.solve_ambiguities(scenario.station_locs, scenario.station_data, sta1, sta2, prn1, prn2, ticks)
#     if res is None:
#         return True
#
#     bad = res[4]    # need to figure out what this is...
#     for i, (n1, n2) in enumerate(res[0]):
#         if members[i].n1:
#             diffs.append(abs(members[i].n1 - n1))
#             diffs.append(abs(members[i].n2 - n2))
#         members[i].n1 = n1
#         members[i].n2 = n2
#         # members[i].n1s.append(n1)
#         # members[i].n2s.append(n2)
#         # members[i].werrs.append( (res[1][i], res[2], res[3]) )
#     return bad
#
# def correct_groups(scenario, groups):
#     '''
#     Runs every group in 'groups' through the ambiguity correction. This appears to take a long time
#     based on the print statements.
#     :param scenario:
#     :param groups:
#     :return:
#     '''
#     bads = 0
#     for i, group in enumerate(groups):
#         if i % 50 == 0:
#             print(i, len(groups), bads)
#         bads += correct_group(scenario.station_locs, scenario.station_data, group)
#     print( numpy.mean(diffs), bads )

def correct_conns(scenario, conns):
    '''
    Runs the initial least-squares ambiguity correction and gets an initial
    estimate of n1, n2 (adds it to the connection object).
    '''
    print("correcting integer ambiguities")
    t0=datetime.datetime.now()
    for i, conn in enumerate(conns):
        if i % 50 == 0:
            print("completed %d/%d" % (i, len(conns)), end="\r")
            # print("completed %d/%d in %s" % (i, len(conns), (datetime.datetime(t0.year, t0.month, t0.day, 0,0,0) +
            #                                  (datetime.datetime.now() - t0)).strftime('%H:%M:%S')), end="\r")
        # If it already has a decent n1 and n2 value, don't bother:
        if (
            conn.n1 is not None
            and not math.isnan(conn.n1)
            and conn.n2 is not None
            and not math.isnan(conn.n2)
        ):
            continue
        # n1, n2 = ambiguity_correct.solve_ambiguity_lsq(
        n1, n2, n21est, n2flt = ambiguity_correct.solve_ambiguity_least_squares_dense(
            scenario,
            conn.station,
            conn.prn,
            conn.ticks
        )
        conn.n1 = n1
        conn.n2 = n2
        conn.n21est = n21est
        conn.n2flt = n2flt

def correct_conns_code(scenario, conns):
    '''
    This is the 'correct_conns' method that is currently active in the main branch. This one calculates the offset
    value specifically but does not compute N1 or N2.

    Uses code phase data to guess the offset without determining n1/n2.
    Parameters
    ----------
    scenario : <pytid.get_data.ScenarioInfo>
    conns : list of Connection objects for the scenario

    Returns : None
        all operations are in place
    -------
    '''
    for i, conn in enumerate(conns):
        if i % 50 == 0:
            print("completed %d/%d" % (i, len(conns)), end="\r")
        # If it already has a decent n1 and n2 value, don't bother (edit: removing that condition)
        # if (
            # conn.n1 is not None
            # and not math.isnan(conn.n1)
            # and conn.n2 is not None
            # and not math.isnan(conn.n2)
            # and conn.offset is not None
        # ):
        #     continue

        # --- If it already has an offset value, skip it ---
        if conn.offset is not None:
            continue

        conn.offset, _, _, mu_code, mu_carrier = ambiguity_correct.offset(
            scenario,
            conn.station,
            conn.prn,
            conn.ticks
        )
        conn.mu_code = mu_code; conn.mu_carrier = mu_carrier;
    print(' '*20)

def empty_factory():
    return None

def make_conn_map(connections):
    '''
    Make an easy-to-use map. We want to go from (station, receiver (PRN), tick) to the connection object it is a
    part of.
    :param connections:
    :return:
    '''
    conn_map = {}
    stations = {conn.station for conn in connections}
    # defaultdict can mess with pickle, which we really want for caching...
    # so do this slightly uglier version
    for station in {conn.station for conn in connections}:
        conn_map[station] = {
            prn: defaultdict(empty_factory) for prn in get_data.satellites
        }

    for conn in connections:
        for tick in conn.ticks:
            conn_map[conn.station][conn.prn][tick] = conn

    return conn_map

# def correct_conns_byo_algorithm(scenario, conns, ac_algo):
#     '''
#     Flexible method to compute the integer ambiguities that allows providing a method to do the
#     computation on the fly. Just for testing.
#
#     Parameters
#     ----------
#     scenario
#     conns
#     ac_algo : function
#         must have signature 'ac_algo(<ScenarioInfo>, station, prn, list of Ticks)'
#
#     Returns
#     -------
#
#     '''
#     print("correcting integer ambiguities with a be-spoke algorithm!")
#     for i, conn in enumerate(conns):
#         if i % 50 == 0:
#             print("completed %d/%d" % (i, len(conns)), end="\r")
#         n1, n2 = ac_algo( scenario, conn.station, conn.prn, conn.ticks )
#         conn.n1 = n1
#         conn.n2 = n2


# def solved_conn_map(dog, station_locs, station_data):
#     '''DEPRECATED 2/24/21
#
#     :param dog: AstroDog object
#     :param station_locs:
#     :param station_data:
#     :return:
#     '''
#     conns = get_connections(dog, station_locs, station_data)
#     groups, unpaired = get_groups(station_data, conns)
#     print(len(unpaired), "unpaired")
#     correct_groups(station_locs, station_data, groups)
#     return make_conn_map(conns)