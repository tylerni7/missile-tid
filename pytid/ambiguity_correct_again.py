"""
correct for integer ambiguities in carrier phase data...

trying this again
"""

import ctypes
from laika.lib import coordinates
from itertools import product
import math
import numpy

import tec

lambda_ws = {}
lambda_ns = {}
lambda_1s = {}
lambda_2s = {}
for ident, freqs in tec.F_lookup.items():
    lambda_ws[ident] = tec.C/(freqs[0] - freqs[1])
    lambda_ns[ident] = tec.C/(freqs[0] + freqs[1])
    lambda_1s[ident] = tec.C/(freqs[0])
    lambda_2s[ident] = tec.C/(freqs[1])

brute = ctypes.CDLL("brute.so")
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
    
    return (v11[0] - v12[0]) - (v21[0] - v22[0])

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

def solve_ambiguities(station_data, sta1, sta2, prn1, prn2, ticks):
    # step 0: remove ticks where we don't have all the data we need...
    # step 1: find widelane double difference to estimate ∆∇N_W
    # step 2: estimate ∆∇Bc and ∆∇N_W to estimate ∆∇N_1, ∆∇N_2
    # step 3: solve N_Ws
    # step 4: use ∆∇N_1, ∆∇N_2, N_Ws and measurements to estimate N1s and N2s

    # step 3: use fixed ∆∇N_1 to get better estimate of ∆∇Bc
    # step 4: use z3 and ∆∇Bc to estimate the actual Bc values
    # step 5: use Bc values to estimate N_1
    # step 6: use N_1 to estimate N_2 using the geometry free
    #   combination

    # initialize wavelengths for this frequency band
    lambda_1 = lambda_1s[prn1[0]]
    lambda_2 = lambda_2s[prn1[0]]
    lambda_n = lambda_ns[prn1[0]]
    lambda_w = lambda_ws[prn1[0]]

    all_ticks = set(ticks[0]) & set(ticks[1]) & set(ticks[2]) & set(ticks[3])

    # step 1: get widelane dd (∆∇N_W)
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

    # step 2:
    # i) estimate ∆∇Bc
    ddbcs = [
        double_difference(
            bias(tec.ionosphere_free),
            station_data, sta1, sta2, prn1, prn2, tick
        ) for tick in all_ticks
    ]
    ddbc = numpy.mean(ddbcs)
    #print("ionosphere free bias dd: {0:0.3f} +/- {1:0.4f}".format(
    #    ddbc, numpy.std(ddbcs)
    #))

    # ii) estimate ∆∇N_1
    ddn1 = (ddbc / lambda_n) - (lambda_w * widelane_dd / lambda_2)
    #print("n1 double difference: {0:0.3f}".format(ddn1))
    ddn1 = round(ddn1)

    # ii) estimate ∆∇N_2 (N_W = N_1 - N_2)
    ddn2 = ddn1 - widelane_dd
    #print("n2 double difference: {0:0.3f}".format(ddn2))

    # step 3:
    # estimate N_Ws
    ws, errs, _ = widelane_solve(widelane_dd, station_data, sta1, sta2, prn1, prn2, all_ticks)
    #print(ws)
    #print(errs)

    return ( geometry_free_solve(ddn1, ddn2, ws, station_data, sta1, sta2, prn1, prn2, ticks) )
