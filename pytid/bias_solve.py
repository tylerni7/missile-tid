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
from cvxopt import matrix, solvers, spmatrix
from laika.lib.coordinates import ecef2geodetic
import numpy

lat_lon_res = 1  # 0.2 degrees
time_res = 60*60   # 20 minutes


class Observation:
    def __init__(self, coi, station, sat, tec, slant):
        self.coi = coi
        self.station = station
        self.sat = sat
        self.tec = tec
        self.slant = slant


def opt_solve(coincidences, measurements, svs, recvs):
    """
    minimize x^T * P * x + q^T * x
    st. Ax = b
    """

    # TODO: encode that TEC values are > 0

    # length of x vector
    n = len(measurements) + len(svs) + len(recvs) + len(coincidences)

    # G is m x n matrix
    m = len(measurements)

    A = spmatrix([], [], [], size=(m, n))
    b = matrix(0.0, (m, 1))

    # for each measurement, we have 
    # 1*error(i)
    Xs = []
    Is = []
    Js = []
    def error(i):
        return i
    
    ssvs = sorted(svs)
    def sat_bias(prn):
        i = ssvs.index(prn)
        return len(measurements) + i
    srecvs = sorted(recvs)
    def recv_bias(recv):
        i = srecvs.index(recv)
        return len(measurements) + len(svs) + i
    scois = sorted(coincidences.keys())
    def coincidence_idx(coi):
        i = scois.index(coi)
        # TODO how to get i?
        return len(measurements) + len(svs) + len(recvs) + i

    for i, measurement in enumerate(measurements):
        A[i, error(i)] = 1
        A[i, sat_bias(measurement.sat)] = 1 # measurement.slant
        A[i, recv_bias(measurement.station)] = 1 # measurement.slant
        A[i, coincidence_idx(measurement.coi)] = 1 / measurement.slant # measurement.slant

        b[i] = measurement.tec / measurement.slant

    P = spmatrix([], [], [], size=(n, n))
    for i, measurement in enumerate(measurements):
        # less weight errors on measurements with high slant factors
        P[i,i] = 10 * (measurement.slant**4)
    for i in range(m, n):
        if i < m + len(svs):
            P[i,i] = 1
        elif i < m + len(svs) + len(recvs):
            P[i,i] = .5
        else:
            P[i,i] = 0
    q = matrix(0.0, (n, 1))

    G = spmatrix([], [], [], size=(m, n))
    h = matrix(0.0, (m, 1))

    sol = solvers.qp(P, q, G, h, A, b)

    errors = {i:sol['x'][error(i)] for i in range(len(measurements))}
    sat_biases =  {prn:sol['x'][sat_bias(prn)] for prn in svs}
    recv_biases = {recv:sol['x'][recv_bias(recv)] for recv in recvs}
    tec_values =  {coi:sol['x'][coincidence_idx(coi)] for coi in coincidences.keys()}

    return errors, sat_biases, recv_biases, tec_values, sol


def round_to_res(num, res):
    return round(num/res)*res

def gather_data(station_vtecs):
    # mapping of (lat, lon, time) to [measurement_idx]
    coincidences = defaultdict(list)
    measurements = []
    svs = set()
    recvs = set()

    for cnt, (station, station_dat) in enumerate(station_vtecs.items()):
        print("gathering data for %3d/%d" % (cnt, len(station_vtecs)))
        for prn, (locs, dats, slants) in station_dat.items():
            for i in range(0, len(locs), time_res//30):
                # look through all data in this time window
                cois = set()
                for j in range(i, i+time_res//30):
                    # skip if there's no data...
                    if locs[j] is None:
                        continue
                    lat, lon, _ = ecef2geodetic(locs[j])
                    coi = (round_to_res(lat, lat_lon_res), round_to_res(lon, lat_lon_res), i)
                    # only log unique COIs
                    if coi in cois:
                        continue
                    cois.add(coi)
                    obs = Observation(coi, station, prn, dats[j], slants[j])

                    coincidences[coi].append(obs)
    
    final_coincidences = dict()
    # only include coincidences with >= 2 measurements
    for coi, obss in coincidences.items():
        if len({obs.station for obs in obss}) > 1:
            final_coincidences[coi] = obss
            for obs in obss:
                svs.add(obs.sat)
                recvs.add(obs.station)
                measurements.append(obs)

    return final_coincidences, measurements, svs, recvs

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