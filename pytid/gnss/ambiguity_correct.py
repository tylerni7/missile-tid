from collections.abc import Iterable
import ctypes
from itertools import product
import functools
import math
import numpy

from pytid.gnss import tec
from pytid.utils.io import find_shared_objects

lambda_ws = {}
lambda_ns = {}
lambda_1s = {}
lambda_2s = {}
lambda_5s = {}
for ident, freqs in tec.F_lookup.items():
    lambda_ws[ident] = tec.C/(freqs[0] - freqs[1])
    lambda_ns[ident] = tec.C/(freqs[0] + freqs[1])
    lambda_1s[ident] = tec.C/(freqs[0])
    lambda_2s[ident] = tec.C/(freqs[1])
    lambda_5s[ident] = tec.C/(freqs[2])

# Find the path to the compiled C libraries. Setup.py must have been run!
so_path = find_shared_objects(prefix="brute.")

brute = ctypes.CDLL(so_path)
brute.brute_force.restype = ctypes.c_double
brute.brute_force.argtypes = [
    ctypes.c_int32, ctypes.c_int32,    # n1 and n2 double differences
    ctypes.c_void_p,                   # list of n1-n2 data
    ctypes.c_double, ctypes.c_double,  # wavelengths
    ctypes.c_void_p,                   # Bi values
    ctypes.c_void_p, ctypes.c_void_p   # output of best n1s and n2s
]
brute.brute_force_harder.restype = ctypes.c_double
brute.brute_force_harder.argtypes = [
    ctypes.c_void_p,                   # list of n1-n2 data
    ctypes.c_double, ctypes.c_double,  # wavelengths
    ctypes.c_void_p,                   # Bi values
    ctypes.c_void_p, ctypes.c_void_p   # output of best n1s and n2s
]
brute.brute_force_dd.restype = ctypes.c_double
brute.brute_force_dd.argtypes = [
    ctypes.c_int32,    # double difference
    ctypes.c_double,   # wavelength
    ctypes.c_void_p,   # bias values
    ctypes.c_void_p,   # output of best ns
]


def double_difference(calculator, station_data, sta1, sta2, prn1, prn2, tick):
    # generic double difference calculator
    v11 = calculator(station_data[sta1][prn1][tick])
    v12 = calculator(station_data[sta1][prn2][tick])
    v21 = calculator(station_data[sta2][prn1][tick])
    v22 = calculator(station_data[sta2][prn2][tick])

    if any([v is None for v in {v11, v12, v21, v22}]):
        return math.nan

    if isinstance(v11, Iterable):
        return (v11[0] - v12[0]) - (v21[0] - v22[0])
    else:
        return (v11 - v12) - (v21 - v22)

def bias(signal):
    def f(meas):
        res = signal(meas)
        return res[0] - res[1], res[-1]
    return f

def dd_solve(dd, vr1s1, vr1s2, vr2s1, vr2s2, wavelength):
    biases = numpy.array([vr1s1, vr1s2, vr2s1, vr2s2], dtype=numpy.double)
    ns = numpy.array([0, 0, 0, 0], dtype=numpy.int32)

    err = brute.brute_force_dd(
        ctypes.c_int32(int(dd)),
        ctypes.c_double(wavelength),
        biases.ctypes.data,
        ns.ctypes.data,
    )
    return ns, err, 0

def widelane_solve(dd, station_data, sta1, sta2, prn1, prn2, ticks):
    lambda_w = lambda_ws[prn1[0]]
    vr1s1s = []
    vr1s2s = []
    vr2s1s = []
    vr2s2s = []
    for tick in ticks:
        vr1s1s.append(tec.melbourne_wubbena(station_data[sta1][prn1][tick])[0])
        vr1s2s.append(tec.melbourne_wubbena(station_data[sta1][prn2][tick])[0])
        vr2s1s.append(tec.melbourne_wubbena(station_data[sta2][prn1][tick])[0])
        vr2s2s.append(tec.melbourne_wubbena(station_data[sta2][prn2][tick])[0])
    vr1s1 = numpy.mean(vr1s1s)
    vr1s2 = numpy.mean(vr1s2s)
    vr2s1 = numpy.mean(vr2s1s)
    vr2s2 = numpy.mean(vr2s2s)
    return dd_solve(dd, vr1s1, vr1s2, vr2s1, vr2s2, lambda_w)


def widelane_ambiguity(station_data, sta1, sta2, prn1, prn2, tick):
    """
    use mw double differences to get
    (ddPhi_w - ddR_n)/lambda_w
    which should be the widelane integer ambiguity
    """

    diff = double_difference(
        tec.melbourne_wubbena,
        station_data, sta1, sta2, prn1, prn2, tick
    )

    if math.isnan(diff):
        return diff

    lambda_w = lambda_ws[station_data[sta1][prn1][tick].prn[0]]
    return diff / lambda_w

def lambda_solve(ddn1, ddn2, ws, station_data, sta1, sta2, prn1, prn2, all_ticks):
    lambda_1 = lambda_1s[prn1[0]]
    lambda_2 = lambda_2s[prn1[0]]

    # Φ_i - R_i = B_i + err  with B_i = b_i + λ_1*N_1 - λ_2*N_2
    B_i = bias(tec.geometry_free)
    Bis = []
    for i, (sta, prn) in enumerate(product([sta1, sta2], [prn1, prn2])):
        B_i_samples = []
        for tick in all_ticks:
            B_i_samples.append( B_i(station_data[sta][prn][tick])[0] )
        #print(numpy.mean(B_i_samples), numpy.std(B_i_samples))
        Bis.append(B_i_samples)

    # Φ - R = B + err with B =

    Q = numpy.cov(Bis[:3])

    y = numpy.array([
        [numpy.mean(Bis[0]) - lambda_2 * ws[0]],
        [numpy.mean(Bis[1]) - lambda_2 * ws[1]],
        [numpy.mean(Bis[2]) - lambda_2 * ws[2]],
        [numpy.mean(Bis[3]) - lambda_2 * ws[3] - ddn1 * (lambda_1 - lambda_2)],
    ])

    A = numpy.array([
        [lambda_1 - lambda_2, 0, 0],
        [0, lambda_1 - lambda_2, 0],
        [0, 0, lambda_1 - lambda_2],
        [lambda_2 - lambda_1, lambda_1 - lambda_2, lambda_1 - lambda_2]
    ])

    a, _, _, _ = numpy.linalg.lstsq(A, y)
    n1s = [
        round(a[0][0]),
        round(a[1][0]),
        round(a[2][0]),
        ddn1 - round(a[0][0]) + round(a[1][0]) + round(a[2][0])
    ]
    ns = [
        (n1s[0], n1s[0] - ws[0]),
        (n1s[1], n1s[1] - ws[1]),
        (n1s[2], n1s[2] - ws[2]),
        (n1s[3], n1s[3] - ws[3]),
    ]
    return ns, ws, 0, 0, 0

def geometry_free_solve(ddn1, ddn2, ws, station_data, sta1, sta2, prn1, prn2, ticks):
    lambda_1 = lambda_1s[prn1[0]]
    lambda_2 = lambda_2s[prn1[0]]

    # Φ_i - R_i = B_i + err  with B_i = b_i + λ_1*N_1 - λ_2*N_2
    B_i = bias(tec.geometry_free)

    Bis = [0, 0, 0, 0]

    for i, (sta, prn) in enumerate(product([sta1, sta2], [prn1, prn2])):
        B_i_samples = []
        for tick in ticks[i]:
            B_i_samples.append( B_i(station_data[sta][prn][tick])[0] )
        #print(numpy.mean(B_i_samples), numpy.std(B_i_samples))
        Bis[i] = numpy.mean(B_i_samples)

    Bis = numpy.array(Bis, dtype=numpy.double)
    ws_ints = numpy.array(ws, dtype=numpy.int32)
    n1s = numpy.array([0, 0, 0, 0], dtype=numpy.int32)
    n2s = numpy.array([0, 0, 0, 0], dtype=numpy.int32)

    err = brute.brute_force(
        ctypes.c_int32(int(ddn1)),
        ctypes.c_int32(int(ddn2)),
        ws_ints.ctypes.data,
        ctypes.c_double(lambda_1),
        ctypes.c_double(lambda_2),
        Bis.ctypes.data,
        n1s.ctypes.data,
        n2s.ctypes.data
    )
    #print(n1s, n2s, err)
    """
    err = brute.brute_force_harder(
        ws_ints.ctypes.data,
        ctypes.c_double(lambda_1),
        ctypes.c_double(lambda_2),
        Bis.ctypes.data,
        n1s.ctypes.data,
        n2s.ctypes.data
    )
    print(n1s, n2s, err)
    """
    return [(n1s[i], n2s[i]) for i in range(4)], ws_ints, 0, 0, 0


def solve_ambiguities(station_locs, station_data, sta1, sta2, prn1, prn2, ticks):
    # initialize wavelengths for this frequency band
    lambda_1 = lambda_1s[prn1[0]]
    lambda_2 = lambda_2s[prn1[0]]
    lambda_n = lambda_ns[prn1[0]]
    lambda_w = lambda_ws[prn1[0]]

    all_ticks = set(ticks[0]) & set(ticks[1]) & set(ticks[2]) & set(ticks[3])

    def rho_dd(tick):
        return (
            rho(station_locs, station_data, sta1, prn1, tick)
            - rho(station_locs, station_data, sta1, prn2, tick)
            - rho(station_locs, station_data, sta2, prn1, tick)
            + rho(station_locs, station_data, sta2, prn2, tick)
        )

    def dd_phi(tick, chan):
        return (
            station_data[sta1][prn1][tick].observables.get(chan, math.nan)
            - station_data[sta1][prn2][tick].observables.get(chan, math.nan)
            - station_data[sta2][prn1][tick].observables.get(chan, math.nan)
            + station_data[sta2][prn2][tick].observables.get(chan, math.nan)
        )

    ddrho = numpy.array([rho_dd(tick) for tick in all_ticks])
    ddphi1 = numpy.array([dd_phi(tick, 'L1C') for tick in all_ticks])
    ddphi2 = numpy.array([dd_phi(tick, 'L2C') for tick in all_ticks])

    ddn1 = round(numpy.mean(ddphi1 - ddrho/lambda_1))
    ddn2 = round(numpy.mean(ddphi2 - ddrho/lambda_2))

    widelane_dds = []

    for tick in all_ticks:
        w = widelane_ambiguity(station_data, sta1, sta2, prn1, prn2, tick)
        if math.isnan(w):
            continue
        widelane_dds.append(w)

    widelane_dd = numpy.mean(widelane_dds)
    #print("wideland double difference: {0:0.3f} +/- {1:0.4f}".format(
    #    widelane_dd, numpy.std(widelane_dds)
    #))
    widelane_dd = round(widelane_dd)

    if abs((ddn1 - ddn2) - widelane_dd) > max(5 * numpy.std(widelane_dds), 3):
        print("divergence for %s-%s %s-%s @ %d" % (sta1, sta2, prn1, prn2, min(all_ticks)))
        print("our dd = %d, widelane_dd = %d" % (ddn1 - ddn2, widelane_dd))
        print("n1,n2,w err %0.2f %0.2f %0.2f\n" % (
            numpy.std(ddphi1 - ddrho/lambda_1),
            numpy.std(ddphi2 - ddrho/lambda_2),
            numpy.std(widelane_dds)
        ))

    ws, errs, _ = widelane_solve(widelane_dd, station_data, sta1, sta2, prn1, prn2, all_ticks)

    return lambda_solve(ddn1, ddn2, ws, station_data, sta1, sta2, prn1, prn2, all_ticks)
#    return geometry_free_solve(ddn1, ddn2, ws, station_data, sta1, sta2, prn1, prn2, ticks)

def solve_ambiguity(station_data, sta, prn, ticks):
    freq_1, freq_2, _ = tec.F_lookup[prn[0]]
    lambda_1 = lambda_1s[prn[0]]
    lambda_2 = lambda_2s[prn[0]]
    def obs(tick, chan):
        return station_data[sta][prn][tick].observables.get(chan, math.nan)

    n21ests = []
    n1ests = []
    for tick in ticks:
        # see GNSS eq 7.31
        n21ests.append(
            (obs(tick, 'L1C') - obs(tick, 'L2C'))
            - (freq_1 - freq_2)/(freq_1 + freq_2) * (
                obs(tick, 'C1C')/lambda_1 + obs(tick, 'C2C')/lambda_2
            )
        )

        n1ests.append(
            obs(tick, 'L1C')
            + 1 / (freq_2**2 - freq_1**2) * (
                (freq_1**2 + freq_2**2) * obs(tick, 'C1C') / lambda_1
                - (2 * freq_1 * freq_2) * obs(tick, 'C2C') / lambda_2
            )
        )

    n1 = round(numpy.mean(n1ests))
    n2 = n1 - round(numpy.mean(n21ests))

    return n1, n2


def rho(station_locs, station_data, station, prn, tick):
    if station_data[station][prn][tick].corrected:
        return numpy.linalg.norm(station_locs[station] - station_data[station][prn][tick].sat_pos_final)
    else:
        return numpy.linalg.norm(station_locs[station] - station_data[station][prn][tick].sat_pos)

def obs_factory(station_data, station_clock_biases, sta, prn):
    freq_1, freq_2, freq_5 = tec.F_lookup[prn[0]]
    def obs(tick, chan):
        res = station_data[sta][prn][tick].observables.get(chan, math.nan)
        if math.isnan(res) and res == 'C2C':
            res = station_data[sta][prn][tick].observables.get('C2P', math.nan)
        clock_err = getattr(station_data[sta][prn][tick], "sat_clock_err")
        clock_err += station_clock_biases[sta][tick]
        if chan[0] == 'C':
            return res + clock_err * tec.C
        elif chan == 'L1C':
            return res + clock_err * freq_1
        elif chan == 'L2C':
            return res + clock_err * freq_2
        elif chan == 'L5C':
            return res + clock_err * freq_5
    return obs


def construct_lsq_matrix(
    distance, obs, freqs, ticks,
    fix_diff=None, single_param=True
):
    """
    Generic function to make LSQ matrices for solving ambiguities

    distance: function from tick to distance parameter
    obs: function from tick, channel to observation
    freqs: the two (or three) frequencies
    ticks: the ticks for which we should get data
    fix_diff: optional int tuple. If present it is the (n1-n2, ) value
        (widelane ambiguity). Or for 3 freq the (n1-n2, n1-n3) values
        if None, don't assume a fixed difference
    single_param: if set, use just one A and B value. Otherwise one per tick

    returns: A, y the matrices for lsq stuff.
    """

    assert 2 <= len(freqs) <= 3, "must specify 2 or 3 frequencies"
    if fix_diff:
        assert len(fix_diff) + 1 == len(freqs), "fixed differences have inconsistent length"

    meas_per_tick = 2 * len(freqs)
    lambdas = [tec.C / f for f in freqs]

    ns_to_find = (len(freqs) if fix_diff is None else 1)
    param_coef = 0 if single_param else 2

    y = numpy.zeros(meas_per_tick * len(ticks))
    A = numpy.zeros((
        meas_per_tick * len(ticks),
        (
            ns_to_find  # how many N values to find
            + 2*(1 if single_param else len(ticks))  # a,b values to find
        )
    ))

    for i, tick in enumerate(ticks):
        # see GNSS hofmann-wellenhof, lichtenegger, wasle eq 5.76 and 7.27
        rho = distance(tick)
        y[i*meas_per_tick + 0] = obs(tick, 'L1C') - rho / lambdas[0]
        y[i*meas_per_tick + 1] = obs(tick, 'L2C') - rho / lambdas[1] + (fix_diff[0] if fix_diff else 0)
        y[i*meas_per_tick + 2] = obs(tick, 'C1C') / lambdas[0] - rho / lambdas[0]
        y[i*meas_per_tick + 3] = obs(tick, 'C2C') / lambdas[1] - rho / lambdas[1]
        if len(freqs) == 3:
            y[i*meas_per_tick + 4] = obs(tick, 'L5C') - rho / lambdas[2] + (fix_diff[1] if fix_diff else 0)
            y[i*meas_per_tick + 5] = obs(tick, 'C5C') / lambdas[2] - rho / lambdas[2]

        # x has the format [n1, ?n2, ?n3, (a, b|a_0, b_0, a_1, ... a_n, b_n)]
        A[i*meas_per_tick + 0][0] = 1
        A[i*meas_per_tick + 0][ns_to_find + param_coef * i] = freqs[0]
        A[i*meas_per_tick + 0][ns_to_find + 1 + param_coef * i] = -1/freqs[0]

        A[i*meas_per_tick + 1][1 if fix_diff is None else 0] = 1
        A[i*meas_per_tick + 1][ns_to_find + param_coef * i] = freqs[1]
        A[i*meas_per_tick + 1][ns_to_find + 1 + param_coef * i] = -1/freqs[1]

        A[i*meas_per_tick + 2][ns_to_find + param_coef * i] = freqs[0]
        A[i*meas_per_tick + 2][ns_to_find + 1 + param_coef * i] = 1/freqs[0]

        A[i*meas_per_tick + 3][ns_to_find + param_coef * i] = freqs[1]
        A[i*meas_per_tick + 3][ns_to_find + 1 + param_coef * i] = 1/freqs[1]

        if len(freqs) == 3:
            A[i*meas_per_tick + 4][2 if fix_diff is None else 0] = 1
            A[i*meas_per_tick + 4][ns_to_find + param_coef * i] = freqs[2]
            A[i*meas_per_tick + 4][ns_to_find + 1 + param_coef * i] = -1/freqs[2]

            A[i*meas_per_tick + 4][ns_to_find + param_coef * i] = freqs[2]
            A[i*meas_per_tick + 4][ns_to_find + 1 + param_coef * i] = 1/freqs[2]

    return A, y


def est_widelane(obs, chans, freqs, ticks):
    ests = []
    lambdas = [tec.C / f for f in freqs]

    for tick in ticks:
        # see GNSS hofmann-wellenhof, lichtenegger, wasle eq 7.31
        ests.append(
            (obs(tick, chans[0][0]) - obs(tick, chans[0][1]))
            - (freqs[0] - freqs[1])/(freqs[0] + freqs[1]) * (
                obs(tick, chans[1][0])/lambdas[0] + obs(tick, chans[1][1])/lambdas[1]
            )
        )
    return round(numpy.mean(ests))


def solve_ambiguity_lsq(station_locs, station_data, station_clock_biases, sta, prn, ticks):
    freq_1, freq_2, freq_5 = tec.F_lookup[prn[0]]

    obs = obs_factory(station_data, station_clock_biases, sta, prn)

    n21 = est_widelane(obs, (('L1C', 'L2C'), ('C1C', 'C2C')), (freq_1, freq_2), ticks)

    distance_func = functools.partial(rho, station_locs, station_data, sta, prn)

    if not math.isnan(obs(ticks[0], 'C5C')):
        use_l5 = True
        n31 = est_widelane(obs, (('L1C', 'L5C'), ('C1C', 'C5C')), (freq_1, freq_5), ticks)
    else:
        use_l5 = False

    A, y = construct_lsq_matrix(
        distance_func,
        obs,
        (freq_1, freq_2, freq_5) if use_l5 else (freq_1, freq_2),
        ticks,
        fix_diff=(n21,n31) if use_l5 else (n21,),
        single_param=True
    )

    a, _, _, _ = numpy.linalg.lstsq(A, y, rcond=None)

    return round(a[0]), round(a[0]) - n21

def solve_dd_ambiguity_lsq(station_locs, station_data, station_clock_biases, group):
    freq_1, freq_2, _ = tec.F_lookup[group.prn1[0]]

    def obs(tick, chan):
        def get_meas_obs(meas):
            res = meas.observables.get(chan, math.nan)
            if math.isnan(res) and chan == 'C2C':
                return meas.observables.get('C2P', math.nan)
            return res
        return group.double_difference(get_meas_obs)

    n21 = est_widelane(obs, (('L1C', 'L2C'), ('C1C', 'C2C')), (freq_1, freq_2), group.ticks)

    # distance doesn't matter for double difference
    distance_func = lambda x:0

    A, y = construct_lsq_matrix(
        distance_func,
        obs,
        (freq_1, freq_2),
        group.ticks,
        #fix_diff=(n21,),
        single_param=True
    )

    a, _, _, _ = numpy.linalg.lstsq(A, y, rcond=None)

    return round(a[0]), round(a[1]) # - n21
    pass