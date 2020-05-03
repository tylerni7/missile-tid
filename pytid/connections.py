"""
each satellite <-> station "connection" has fixed values for
integer ambiguities. Break this out into an easy-to-use class
"""
from collections import defaultdict
import math
import numpy

from laika import helpers

#import ambiguity_correct_swap_fix as ambiguity_correct
import ambiguity_correct_rho as ambiguity_correct
import tec

CYCLE_SLIP_CUTOFF = 6
MIN_CON_LENGTH = 20  # 10 minutes worth of connection
DISCON_TIME = 5      # cycle slip for >= 5 samples of no info

EL_CUTOFF = 0.30

def contains_needed_info(measurement):
    observable = measurement.observables
    chan2 = 'C2P' if not math.isnan(observable['C2P']) else 'C2C'
    needed = {'L1C', 'L2C', 'C1C', chan2}
    for need in needed:
        if math.isnan(observable.get(need, math.nan)):
            return False
    return True

class Connection:
    def __init__(self, dog, station_locs, station_data, station, prn, tick0, tickn):
        self.dog = dog
        self.loc = station_locs[station]
        self.this_data = station_data[station][prn]
        self.station = station
        self.prn = prn
        self.tick0 = tick0
        self.tickn = tickn
        self.ticks = None

        # integer ambiguities... what we really want
        self.n1 = None
        self.n2 = None

        self.n1s = []
        self.n2s = []
        self.werrs = []

        # TODO: this could incorporate phase windup as well?

        self.filter_ticks()
#        self.detect_slips()

    def filter_ticks(self):
        if self.ticks is not None:
            return
        
        self.ticks = []
        last_seen = self.tick0
        last_mw = None
        for tick in range(self.tick0, self.tickn):
            if not self.this_data[tick]:
                continue

            if not contains_needed_info(self.this_data[tick]):
                continue

            # ignore "bad connections" (low elevation)
            if not self.this_data[tick].processed:
                if not self.this_data[tick].process(self.dog):
                    continue
            if not self.this_data[tick].corrected:
                self.this_data[tick].correct(self.loc, self.dog)
            el, _ = helpers.get_el_az(self.loc, self.this_data[tick].sat_pos)
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
            mw, _ = tec.melbourne_wubbena(self.this_data[tick])
            if last_mw is not None and abs(last_mw - mw) > CYCLE_SLIP_CUTOFF:
                print("cycle slip: {0}-{1}@{2} jump of {3:0.2f}".format(
                    self.station, self.prn, tick, last_mw - mw
                ))
                break
            last_mw = mw

            # all good
            self.ticks.append(tick)
        
        if self.ticks:
            self.tick0 = self.ticks[0]
            self.tickn = self.ticks[-1]
        else:
            self.ticks = [tick]
            self.tick0 = tick
            self.tickn = tick

    def detect_slips(self):
        """
        if we get a cycle slip, our ambiguity numbers are invalid
        so it becomes a new connection.
        Easy way to detect is the Melbourne Wubbena combination
        """
        past = None
        last_tick = None
        for i, tick in enumerate(self.ticks):
            mw, _ = tec.melbourne_wubbena(self.this_data[tick])
            if past is None:
                past = mw
            elif abs(past - mw) > CYCLE_SLIP_CUTOFF:
                print("cycle slip: {0}-{1} {2} jump of {3:0.2f}".format(
                    self.station, self.prn, tick, past - mw
                ))
                last_tick = i
                break
        
        # cut off data if need be
        if last_tick is not None:
            self.ticks = self.ticks[:last_tick]
            if self.ticks:
                self.tickn = self.ticks[-1]

def make_connections(dog, station_locs, station_data, station, prn, tick0, tickn):
    """
    given data and a bunch of ticks, split it up into connections
    """
    def next_valid_tick(tick):
        for i in range(tick + 1, tickn):
            if station_data[station][prn][i]:
                return i
        return tickn

    connections = []
    ticki = tick0
    while ticki < tickn:
        con = Connection(dog, station_locs, station_data, station, prn, ticki, tickn)
        if not con.ticks:
            ticki = next_valid_tick(ticki)
            continue
        elif len(con.ticks) > MIN_CON_LENGTH:
            connections.append(con)
        ticki = next_valid_tick(con.tickn)
    return connections

def get_connections(dog, station_locs, station_data):
    connections = []
    for station in station_data.keys():
        for prn in station_data[station].keys():
            ticks = station_data[station][prn].keys()
            if not ticks:
                continue
            connections += make_connections(
                dog,
                station_locs,
                station_data,
                station,
                prn,
                min(ticks),
                max(ticks)
            )
    return connections

def get_groups(connections):
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
            groups.append( partners | set([con]) )
            unpaired -= partners
            unpaired.remove(con)
    
    return groups, unpaired

diffs = []

def correct_group(station_locs, station_data, group):
    members = sorted(group, key=lambda x:(x.station, x.prn))
    sta1 = members[0].station
    sta2 = members[2].station
    prn1 = members[0].prn
    prn2 = members[1].prn

    ticks = [members[i].ticks for i in range(4)]
    res = ambiguity_correct.solve_ambiguities(station_locs, station_data, sta1, sta2, prn1, prn2, ticks)
    if res is None:
        return True

    bad = res[4]
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

def correct_groups(station_locs, station_data, groups):
    bads = 0
    for i, group in enumerate(groups):
        if i % 50 == 0:
            print(i, len(groups), bads)
        bads += correct_group(station_locs, station_data, group)
    print( numpy.mean(diffs), bads )

def make_conn_map(connections):
    # make an easy-to-use map
    conn_map = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : None)))
    for conn in connections:
        for tick in conn.ticks:
            conn_map[conn.station][conn.prn][tick] = conn
    
    return conn_map

def solved_conn_map(dog, station_locs, station_data):
    conns = get_connections(dog, station_locs, station_data)
    groups, unpaired = get_groups(conns)
    print(len(unpaired), "unpaired")
    correct_groups(station_locs, station_data, groups)
    return make_conn_map(conns)