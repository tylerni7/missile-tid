"""
each satellite <-> station "connection" has fixed values for
integer ambiguities. Break this out into an easy-to-use class
"""
from collections import defaultdict
from laika import helpers
from laika.lib import coordinates
import math
import numpy

from pytid.gnss import ambiguity_correct
from pytid.gnss import get_data
from pytid.gnss import tec

CYCLE_SLIP_CUTOFF_MW = 10
CYCLE_SLIP_CUTOFF_DELAY = 0.3
MIN_CON_LENGTH = 20  # 10 minutes worth of connection
DISCON_TIME = 5      # cycle slip for >= 5 samples of no info
FULL_CORRECTION = False  # whether to do the full laika "correct" method, largely unnecessary
EL_CUTOFF = 0.30


def contains_needed_info(measurement):
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    needed = {'L1C', 'L2C', 'C1C', chan2}
    for need in needed:
        if math.isnan(observable.get(need, math.nan)):
            return False
    return True


class Connection:
    def __init__(self, scenario, station, prn, tick0, tickn, filter_ticks=True):
        self.scenario = scenario
        self.loc = scenario.station_locs[station]
        self.station = station
        self.prn = prn
        self.tick0 = tick0
        self.tickn = tickn
        self.ticks = None

        # integer ambiguities... what we really want
        self.n1 = None
        self.n2 = None

        # "raw" offset: the difference from the code phase tec values
        self.offset = None

        # testing, place to hold n_values calculated in various ways
        self.n12_repo = {}

        self.n1s = []
        self.n2s = []
        self.werrs = []

        # TODO: this could incorporate phase windup as well?
        if filter_ticks:
            self.filter_ticks(scenario.station_data[station][prn])

    def filter_ticks(self, this_data):
        '''
        Starts with an assumed 'connection' of length zero at the first time point ('tick'). Runs through the list
        of ticks (integers) in increasing order. Skips ticks where the satelite elevation was too low. With each tick,
        judges whether the connection has 'dropped' either due to a) DISCONNECTION, or b) CYCLE_SLIP. If it has dropped,
        it stops counting and the connection is considered 'complete' (i.e. initialized).
        :param this_data:
        :return:
        '''
        if self.ticks is not None:
            return

        self.ticks = []
        last_seen = self.tick0
        last_mw = None
        last_tec = None
        for tick in range(self.tick0, self.tickn):
            if not this_data[tick]:
                continue

            if not contains_needed_info(this_data[tick]):
                continue

            # ignore "bad connections" (low elevation)
            if not this_data[tick].processed:
                if not this_data[tick].process(self.scenario.dog):
                    continue

            if FULL_CORRECTION and not this_data[tick].corrected:
                this_data[tick].correct(self.loc, self.scenario.dog)

            el = self.scenario.station_el(self.station, this_data[tick].sat_pos)
            if el < EL_CUTOFF:
                continue

            # detect slips due to long disconnection time
            if tick - last_seen > DISCON_TIME:
#                print("cycle slip: {0}-{1}@{2} exceeded discon time ({3})".format(
#                    self.station, self.prn, tick, last_seen
#                ))
                break
            last_seen = tick

            # detect cycle slips due to changing N_w
            mw, _ = tec.melbourne_wubbena(self.scenario.dog, this_data[tick])
            if last_mw is not None and abs(last_mw - mw) > CYCLE_SLIP_CUTOFF_MW:
                print("cycle slip: {0}-{1}@{2} mw jump of {3:0.2f}".format(
                    self.station, self.prn, tick, last_mw - mw
                ))
                break
            last_mw = mw
            # OR cycle slips from big changes in TEC
            delay, _ = tec.calc_carrier_delay(self.scenario.dog, this_data[tick])
            if last_tec is not None and abs(last_tec - delay) > CYCLE_SLIP_CUTOFF_DELAY:
                print("cycle slip: {0}-{1}@{2} delay jump of {3:0.2f}".format(
                    self.station, self.prn, tick, last_tec - delay
                ))
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

    def remove_missing_sat_clock_err_ticks(self, scenario):
        '''
        For some ticks in a connection the sat_clock_error might be missing from laika.
        If that is the case, just boot that tick from the collection.
        '''
        tind_offset = 0
        for tind in range(len(self.ticks)):
            t=self.ticks[tind-tind_offset]
            if math.isnan(scenario.station_data[self.station][self.prn][t].sat_clock_err):
                tout = self.ticks.pop(tind-tind_offset)
                self.ticks_missing_sat_clock_err.append(tout)
                tind_offset+=1
        if len(self.ticks)>0:
            self.tick0 = self.ticks[0]
            self.tickn = self.ticks[-1]

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


def make_connections(scenario, station, prn, tick0, tickn):
    """
    given data and a bunch of ticks, split it up into connections
    """
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

def get_connections(scenario, skip=None):
    '''
    For each (station,satelite), grabs the ticks and GNSS measurements (the innermost dict of station_data) and
    runs it through "make_connections()" to get a list of Connection objects.
    :param scenario: ScenarioInfo object
    :param skip: a list of stations that should be skipped.
    :return:
    '''
    connections = []
    for station in scenario.stations:
        if skip and station in skip:
            continue
        for prn in scenario.station_data[station].keys():
            ticks = scenario.station_data[station][prn].keys()
            if not ticks:
                continue
            connections += make_connections(
                scenario,
                station,
                prn,
                min(ticks),
                max(ticks)
            )
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

def correct_group(scenario, group):
    '''
    Applies the ambiguity correction calculated from a 'group' of station/prns.
    :param scenario:
    :param group:
    :return:
    '''
    members = sorted(group, key=lambda x:(x.station, x.prn))
    sta1 = members[0].station
    sta2 = members[2].station
    prn1 = members[0].prn
    prn2 = members[1].prn

    ticks = [members[i].ticks for i in range(4)]    # list of lists
    res = ambiguity_correct.solve_ambiguities(scenario.station_locs, scenario.station_data, sta1, sta2, prn1, prn2, ticks)
    if res is None:
        return True

    bad = res[4]    # need to figure out what this is...
    for i, (n1, n2) in enumerate(res[0]):
        if members[i].n1:
            diffs.append(abs(members[i].n1 - n1))
            diffs.append(abs(members[i].n2 - n2))
        members[i].n1 = n1
        members[i].n2 = n2
        members[i].n1s.append(n1)
        members[i].n2s.append(n2)
        members[i].werrs.append( (res[1][i], res[2], res[3]) )
    return bad

def correct_groups(scenario, groups):
    '''
    Runs every group in 'groups' through the ambiguity correction. This appears to take a long time
    based on the print statements.
    :param scenario:
    :param groups:
    :return:
    '''
    bads = 0
    for i, group in enumerate(groups):
        if i % 50 == 0:
            print(i, len(groups), bads)
        bads += correct_group(scenario.station_locs, scenario.station_data, group)
    print( numpy.mean(diffs), bads )

def correct_conns(scenario, conns):
    '''
    Runs the initial least-squares ambiguity correction and gets an initial
    estimate of n1, n2 (adds it to the connection object).
    :param scenario:
    :param conns:
    :return:
    '''
    print("correcting integer ambiguities")
    for i, conn in enumerate(conns):
        if i % 50 == 0:
            print("completed %d/%d" % (i, len(conns)), end="\r")
        # If it already has a decent n1 and n2 value, don't bother:
        if (
            conn.n1 is not None
            and not math.isnan(conn.n1)
            and conn.n2 is not None
            and not math.isnan(conn.n2)
        ):
            continue
        n1, n2 = ambiguity_correct.solve_ambiguity_lsq(
            scenario,
            conn.station,
            conn.prn,
            conn.ticks
        )
        conn.n1 = n1
        conn.n2 = n2


def correct_conns_code(scenario, conns):
    '''
    Uses code phase data to guess the offset without determining n1/n2
    :param scenario:
    :param conns:
    :return:
    '''
    for i, conn in enumerate(conns):
        if i % 50 == 0:
            print("completed %d/%d" % (i, len(conns)), end="\r")
        # If it already has a decent n1 and n2 value, don't bother:
        if (
            conn.n1 is not None
            and not math.isnan(conn.n1)
            and conn.n2 is not None
            and not math.isnan(conn.n2)
            and conn.offset is not None
        ):
            continue

        conn.offset, _ = ambiguity_correct.offset(
            scenario,
            conn.station,
            conn.prn,
            conn.ticks
        )


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

def solved_conn_map(dog, station_locs, station_data):
    '''

    :param dog: AstroDog object
    :param station_locs:
    :param station_data:
    :return:
    '''
    conns = get_connections(dog, station_locs, station_data)
    groups, unpaired = get_groups(station_data, conns)
    print(len(unpaired), "unpaired")
    correct_groups(station_locs, station_data, groups)
    return make_conn_map(conns)