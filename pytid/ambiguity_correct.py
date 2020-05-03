"""
correct for integer ambiguities in carrier phase data...
"""

import ctypes
from laika.lib import coordinates
from itertools import product
import math
import numpy
from z3 import Solver, Optimize, Ints, Real, Reals, ToReal, sat, unsat

import tec
import tropo

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


def phi_double_difference(station_data, sta1, sta2, prn1, prn2, tick):
    # has to be same frequency bands for this to work
    assert prn1[0] == prn2[0]

    return double_difference(
        tec.calc_carrier_delay,
        station_data, sta1, sta2, prn1, prn2, tick
    )


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


def dd_solve(dd, vr1s1, vr1s2, vr2s1, vr2s2, wavelength):
    biases = numpy.array([vr1s1, vr1s2, vr2s1, vr2s2], dtype=numpy.double)
    ns = numpy.array([0, 0, 0, 0], dtype=numpy.int32)

    err = brute.brute_force_dd(
        ctypes.c_int32(int(dd)),
        ctypes.c_double(wavelength),
        biases.ctypes.data,
        ns.ctypes.data,
    )
    return ns, err

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


def estimate_Bc(meas):
#    meas = station_data[sta][prn][tick]
    phase, pseudorange, wavelength = tec.ionosphere_free(meas)
    return phase - pseudorange, wavelength

def bias(signal):
    def f(meas):
        res = signal(meas)
        return res[0] - res[1], res[-1]
    return f


def test_n1(N_w, station_data, sta, prn, ticks):
    # given N_w = N_1 - N_2
    # test out a N1/N2 combinations

    # get things that don't change base on N_1:
    #   using N_w get b_w, delay_factor_w, B_c, B_i, wavelengths
    # for each N_1 candidate:
    #   using N_w, N_1_candidate, B_c, estimate b_c, use that to get
    #      a b_I estimate plug in to get ERR_1
    #   using b_c, b_w, N_1_candidate, delay_factor_w, estimate b_I
    #      and use that to get ERR_1

    lambda_1 = lambda_1s[prn[0]]
    lambda_2 = lambda_2s[prn[0]]
    lambda_n = lambda_ns[prn[0]]
    lambda_w = lambda_ws[prn[0]]

    b_w = numpy.mean([
        tec.melbourne_wubbena(station_data[sta][prn][tick]) for tick in ticks
    ]) - lambda_w * N_w

    freqs = tec.F_lookup[prn[0]]
    # TODO why are f1 and f2 reversed from what I expect?
    delay_factor_w = freqs[0]*freqs[1]/(freqs[1]**2 - freqs[0]**2)
    B_c = numpy.mean([
        estimate_Bc(station_data[sta][prn][tick])[0] for tick in ticks
    ])
    gf_bias = bias(tec.geometry_free)
    B_i = numpy.mean([
        gf_bias(station_data[sta][prn][tick]) for tick in ticks
    ])


    N_1_best = None
    err_best = 10000

    for N_1_cand in range(-200, 200):
        N_2_cand = N_1_cand - N_w

        b_c_meas = B_c - lambda_n * (N_1_cand + N_w * lambda_w / lambda_2)
        b_i_meas = B_i - lambda_1 * N_1_cand + lambda_2 * N_2_cand

        b_i_est = (b_w - b_c_meas) / delay_factor_w
        b_c_est = b_w - delay_factor_w * b_i_meas
        
        err1 = (b_i_est - b_i_meas)**2
        err2 = (b_c_est - b_c_meas)**2
        if err1 + err2 < err_best:
            err_best = err1 + err2
            N_1_best = N_1_cand
    return N_1_best, err_best


def solve_ambiguities(station_data, sta1, sta2, prn1, prn2, ticks):
    """
    Attempt to solve integer ambiguities for the given stations x prns
    on the given ticks.

    TODO: this doesn't use a ∆∇B_c estimate to get ∆∇N_1, because I did
        not have luck with that. It could help improve results...
    """

    # step 1: find widelane double difference to estimate ∆∇N_W
    # step 2: solve N_Ws
    # step 3: use N_Ws to solve for probable N_1s

    # initialize wavelengths for this frequency band
    lambda_1 = lambda_1s[prn1[0]]
    lambda_2 = lambda_2s[prn1[0]]
    lambda_n = lambda_ns[prn1[0]]
    lambda_w = lambda_ws[prn1[0]]

    # step 1: get widelane dd (∆∇N_W)
    widelane_dds = []

    for tick in ticks:
        w = widelane_ambiguity(station_data, sta1, sta2, prn1, prn2, tick)
        if math.isnan(w):
            continue
        widelane_dds.append(w)

    widelane_dd = numpy.mean(widelane_dds)
#    print("wideland double difference: {0:0.3f} +/- {1:0.4f}".format(
#        widelane_dd, numpy.std(widelane_dds)
#    ))

    widelane_dd = round(widelane_dd)

    # step 2:
    # estimate N_Ws
    ws, err = widelane_solve(widelane_dd, station_data, sta1, sta2, prn1, prn2, ticks)
    if err >= 10000.0:
        print("couldn't find solution for ", sta1, sta2, prn1, prn2)
        return None

    # step 3:
    ns = []
    for i, (sta, prn) in enumerate(product([sta1, sta2], [prn1, prn2])):
        n1, err = test_n1(ws[i], station_data, sta, prn, ticks)
        if err >= 10000.0:
            print("couldn't find solution for ", sta1, sta2, prn1, prn2)
            return None
        n2 = n1 - ws[i]
        ns.append( (n1, n2, err) )
    
    return ns
