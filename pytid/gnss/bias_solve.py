"""
calculate receiver and satellite biases...

general approach is:
    1) get measurements of (vTEC + unknowns, station, PRN, lat, lon, time)
    2) find several measurements that agree on (lat, lon, time)
    3) assume vTEC for those measurements are fixed
    4) solve for all the unknowns...

slant_factor(i) * (p1 - p2) = delay(i) = slant_factor(i) * TEC + Offset(station) + Offset(satellite)

we will end up with lots of equations like

slant_factor(i)*TEC(lat, lon, time) + Offset(station) + Offset(satellite) + error(i) = delay(i)

and minimize error(i)**2

We can use quadratic programming for this...
"""

from collections import defaultdict
from datetime import timedelta
from laika.lib.coordinates import ecef2geodetic
from laika.gps_time import GPSTime
import numpy
from operator import is_

from scipy import optimize, sparse

from .tec import K, F_lookup

# same res as the published TEC maps so we can compare more easily
lat_res = 2.5      # 2.5 degrees 
lon_res = 5.0      # 5.0 degrees
time_res = 60*15   # 15 minutes


class Observation:
    def __init__(self, coi, station, sat, tec, slant, gpstime):
        self.coi = coi
        self.station = station
        self.sat = sat
        self.tec = tec
        self.slant = slant
        self.gpstime = gpstime
    
    def __eq__(self, other):
        return (
            self.coi == other.coi
            and self.station == other.station
            and self.sat == other.sat
            and self.tec == other.tec
            and self.slant == other.slant
            and self.gpstime == other.gpstime
        )

def lsq_solve(dog, coincidences, measurements, svs, recvs, sat_biases=None, ionmap=None):
    '''
    Uses connection combinations that form a coincidence to calculate the biases for the sats and receivers.
    (should probably put a reference in here).
    :param coincidences:
    :param measurements:
    :param svs:
    :param recvs:
    :param sat_biases:
    :return:
    '''

    ssvs = sorted(svs)
    def sat_bias(prn):
        i = ssvs.index(prn)
        return i
    
    # different satellite constellations
    constellations = sorted({prn[0] for prn in svs})
    constellation_params = len(constellations)
    if 'R' in constellations:
        # FDMA constellation isn't just a single bias: we'll model as
        # a single bias PLUS a component linear in frequency
        # why oh why did they use FDMA????
        constellation_params += 1
    
    srecvs = sorted(recvs)
    def recv_bias(recv, prn):
        # each constellation can have different signal paths and therefore
        # different instrumentation biases
        i = srecvs.index(recv)
        j = constellations.index(prn[0])
        return len(svs) + i * constellation_params + j
    scois = sorted(coincidences.keys())
    def coincidence_idx(coi):
        i = scois.index(coi)
        # TODO how to get i?
        return len(svs) + len(recvs) * len(constellations) + i

    print("constructing matrix")
    # total unknowns to recover
    n = len(svs) + len(recvs) * constellation_params + len(coincidences)
    # each measurement we have
    m = len(measurements)
    A = sparse.dok_matrix((m, n))
    b = numpy.zeros((m, ))

    for i, measurement in enumerate(measurements):
        # target term: the stec (vtec/slant) value we measured
        b[i] = measurement.tec / measurement.slant

        # sat bias term
        if not sat_biases or measurement.sat not in sat_biases:
            # sat biases have opposite sign by convention...
            A[i, sat_bias(measurement.sat)] = -1
        else:
            # remove known bias
            b[i] -= -sat_biases[measurement.sat]


        # receiver bias term
        A[i, recv_bias(measurement.station, measurement.sat)] = 1

        # linear terms for FDMA biases on GLONASS
        if measurement.sat[0] == 'R':
            A[i, recv_bias(measurement.station, measurement.sat) + 1] = (
                dog.get_glonass_channel(measurement.sat, measurement.gpstime)
            )

        if not ionmap or measurement.coi not in ionmap:
            A[i, coincidence_idx(measurement.coi)] = 1 / measurement.slant
        else:
            # remove known TEC
            b[i] -= ionmap[measurement.coi] / measurement.slant


    print("solving")
    upper_bounds = numpy.ones(n) * numpy.inf
    lower_bounds = numpy.concatenate((
        numpy.ones(len(svs) + len(recvs) * constellation_params) * -numpy.inf,
        numpy.zeros(len(coincidences))
    ))
    res = optimize.lsq_linear(A, b, bounds=(lower_bounds, upper_bounds))
    sat_biases =  {prn:res.x[sat_bias(prn)] for prn in svs}
    recv_biases = {
        recv:res.x[
            recv_bias(recv, 'G01') : recv_bias(recv, 'G01') + constellation_params
        ] for recv in recvs
    }
    tec_values =  {coi:res.x[coincidence_idx(coi)] for coi in coincidences.keys()}
    return sat_biases, recv_biases, tec_values


def round_to_res(num, res):
    return round(num/res)*res

def gather_data(start_time, station_vtecs):
    '''
    Looks for 'coincidences'. A 'coincidence' is a set of observables for various sat/rec pairs that cross
    into the ionosphere at approximately the same location (lat,lon).

    Returns four items:
        final_coicidences (dict) {(lat, lon, i): Observation((lat, lon, i), station, prn, vtec, s_to_v, ...}
        measurements (list): [Observation 1, ....]
        sats, recvrs (set): sets of satellite and receiver objects included.
    '''
    # mapping of (lat, lon, time) to [measurement_idx]
    coincidences = defaultdict(list)
    measurements = []
    svs = set()
    recvs = set()

    for cnt, (station, station_dat) in enumerate(station_vtecs.items()):
        print("gathering data for %3d/%d" % (cnt, len(station_vtecs)))
        for prn, (locs, dats, slants) in station_dat.items():
            if len(locs) == 0:
                continue
            # convert locs to lat lon in bulk for much better speed
            # dirty hack to force numpy to treat the array as 1d
            locs.append(None)
            # indices with locations set
            idxs = numpy.where(numpy.logical_not(numpy.vectorize(is_)(locs, None)))
            locs.pop(-1)
            if len(idxs[0]) == 0:
                continue
            locs_lla = ecef2geodetic(numpy.stack(numpy.array(locs)[idxs]))

            prev_coi = None
            for i, idx in enumerate(idxs[0]):
                cois = set()
                lat, lon, _ = locs_lla[i]
                coi = (
                    round_to_res(lat, lat_res),
                    round_to_res(lon, lon_res),
                    (idx // (time_res / 30)) * 30
                )
                if coi == prev_coi:
                    continue
                prev_coi = coi

                gpstime = start_time + timedelta(seconds=30 * int(idx))
                gpstime = GPSTime.from_datetime(gpstime)
                obs = Observation(coi, station, prn, dats[idx], slants[idx], gpstime)
                coincidences[coi].append(obs)
    
    final_coincidences = dict()
    # only include coincidences with >= 2 measurements
    for coi, obss in coincidences.items():
        if (
            len({obs.station for obs in obss}) > 1
            or len({obs.sat for obs in obss}) > 1
        ):
            final_coincidences[coi] = obss
            for obs in obss:
                svs.add(obs.sat)
                recvs.add(obs.station)
                measurements.append(obs)

    return final_coincidences, measurements, svs, recvs


#
# ***TODO: These functions are not currently used***
#
def remove_bad_measurements(coincidences, measurements, svs, recvs, errors, cutoff=0.8):
    error_threshold = numpy.quantile(numpy.array([abs(err) for i,err in errors.items()]), cutoff)
    bad_obs = {measurements[i] for i,err in errors.items() if abs(err) > error_threshold}

    final_measurements = []
    final_svs = set()
    final_recvs = set()
    final_coincidences = dict()
    # only include coincidences with >= 2 measurements
    # TODO there is a BUG where we stop deleting at some point???
    for coi, obss in coincidences.items():
        if len({obs.station for obs in obss if obs not in bad_obs}) > 1:
            final_coincidences[coi] = obss
            for obs in obss:
                if obs in bad_obs:
                    continue
                final_svs.add(obs.sat)
                final_recvs.add(obs.station)
                final_measurements.append(obs)

    return final_coincidences, final_measurements, final_svs, final_recvs

def solve_until(coincidences, measurements, svs, recvs, error_thresh=5.0):
    solved = opt_solve(coincidences, measurements, svs, recvs)

    def mean_error():
        return numpy.array([abs(err) for i,err in solved[0].items()]).mean()
    
    while mean_error() > error_thresh:
        try:
            print("coincidences: %d" % len(solved[0]))
            print("avg error: %0.1f" % mean_error())
            # remove bad samples
            coincidences, measurements, svs, recvs = remove_bad_measurements(
                                                        coincidences,
                                                        measurements,
                                                        svs,
                                                        recvs,
                                                        solved[0]
                                                    )
            solved = opt_solve(coincidences, measurements, svs, recvs)
        except KeyboardInterrupt:
            return solved, (coincidences, measurements, svs, recvs)
    return solved, (coincidences, measurements, svs, recvs)