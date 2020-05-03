"""
correct for integer ambiguities in carrier phase data...

This file contains stuff I don't want to delete in case it's useful later,
but which I tried and found not to work/be necessary
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



def frac_to_float(frac):
    # helper function for z3
    return (
        frac.numerator_as_long()
        / frac.denominator_as_long()
    )

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

def dd_solve_(dd, vr1s1, vr1s2, vr2s1, vr2s2, wavelength):
    sol = Solver()
    r1s1, r1s2, r2s1, r2s2 = Ints('r1s1 r1s2 r2s1 r2s2')
    err = Real('err')
    err1, err1, err3, err4 = Reals('err1 err2 err3 err4')
#    sol.add(err > 0)

    sol.add(r1s1 - r1s2 - r2s1 + r2s2 == dd)

    sol.add(ToReal(r1s1)*wavelength + err1 > vr1s1)
    sol.add(ToReal(r1s1)*wavelength - err1 < vr1s1)

    sol.add(ToReal(r1s2)*wavelength + err2 > vr1s2)
    sol.add(ToReal(r1s2)*wavelength - err2 < vr1s2)

    sol.add(ToReal(r2s1)*wavelength + err3 > vr2s1)
    sol.add(ToReal(r2s1)*wavelength - err3 < vr2s1)

    sol.add(ToReal(r2s2)*wavelength + err4 > vr2s2)
    sol.add(ToReal(r2s2)*wavelength - err4 < vr2s2)

    if sol.check() != sat:
        return None
    
    def minimize():
        # try to push the error lower, if possible
        for mult in [0.5, 0.85]:
            while sol.check() == sat:
                sol.push()
                sol.check()
                err_bound = frac_to_float(sol.model()[err])
                if err_bound < 0.2:
                    # not gonna do better than that...
                    return
                sol.add(err < err_bound*mult)
            sol.pop()
            sol.check()
    
    minimize()
    return (
        [sol.model()[r].as_long() for r in [r1s1, r1s2, r2s1, r2s2]],
        frac_to_float(sol.model()[err])
    )

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

def dd_solve_(dd, vr1s1, vr1s2, vr2s1, vr2s2, wavelength, ionosphere=False):
    sol = Optimize()
    r1s1, r1s2, r2s1, r2s2 = Ints('r1s1 r1s2 r2s1 r2s2')
#    err = Real('err')
    err1, err2, err3, err4 = Reals('err1 err2 err3 err4')
#    sol.add(err > 0)

    if ionosphere:
        ion = Real('ion')
        sol.add(ion > 0)
        sol.add(ion < 25)
    else:
        ion = 0

    sol.add(r1s1 - r1s2 - r2s1 + r2s2 == dd)

    sol.add(ToReal(r1s1)*wavelength + err1 > vr1s1 - ion)
    sol.add(ToReal(r1s1)*wavelength - err1 < vr1s1 - ion)

    sol.add(ToReal(r1s2)*wavelength + err2 > vr1s2 - ion)
    sol.add(ToReal(r1s2)*wavelength - err2 < vr1s2 - ion)

    sol.add(ToReal(r2s1)*wavelength + err3 > vr2s1 - ion)
    sol.add(ToReal(r2s1)*wavelength - err3 < vr2s1 - ion)

    sol.add(ToReal(r2s2)*wavelength + err4 > vr2s2 - ion)
    sol.add(ToReal(r2s2)*wavelength - err4 < vr2s2 - ion)

    objective = sol.minimize(err1 + err2 + err3 + err4)

    if sol.check() != sat:
        return None
    
    sol.lower(objective)
    if sol.check() != sat:
        return None

    return (
        [sol.model()[r].as_long() for r in [r1s1, r1s2, r2s1, r2s2]],
        [frac_to_float(sol.model()[err]) for err in [err1, err2, err3, err4]],
        frac_to_float(sol.model()[ion]) if ionosphere else 0
    )

def widelane_solve(dd, station_data, sta1, sta2, prn1, prn2, tick):
    vr1s1, lambda_w = tec.melbourne_wubbena(station_data[sta1][prn1][tick])
    vr1s2, _ = tec.melbourne_wubbena(station_data[sta1][prn2][tick])
    vr2s1, _ = tec.melbourne_wubbena(station_data[sta2][prn1][tick])
    vr2s2, _ = tec.melbourne_wubbena(station_data[sta2][prn2][tick])
    return dd_solve(dd, vr1s1, vr1s2, vr2s1, vr2s2, lambda_w)

def l1_ambiguity(station_data, sta1, sta2, prn1, prn2, tick):
    """
    use widelane ambiguity and ionosphere free stuff
    to try to get l1 integer ambiguity
    """

    # first estimate ionosphere free bias double difference
    def ionosphere_free_bias(*args):
        ret = tec.ionosphere_free(*args)
        if ret is not None:
            return ret[1] - ret[0], None
    
    ddbc = double_difference(
        ionosphere_free_bias,
        station_data, sta1, sta2, prn1, prn2, tick
    )

    nw = widelane_ambiguity(station_data, sta1, sta2, prn1, prn2, tick)
    if not math.isnan(nw):
        nw = round(nw)

    lambda_w = lambda_ws[station_data[sta1][prn1][tick].prn[0]]
    lambda_n = lambda_ns[station_data[sta1][prn1][tick].prn[0]]
    lambda_2 = lambda_2s[station_data[sta1][prn1][tick].prn[0]]

    return ddbc, ddbc / lambda_n  - nw * lambda_w / lambda_2


# earth's gravitational constant, this is slightly off in laika...
mu = 398600441800000
def shapiro_cor(station_loc, sat_loc):
    """
    general relativitistic correction factor
    """
    station_r = numpy.linalg.norm(station_loc)
    sat_r = numpy.linalg.norm(sat_loc)
    distance_r = numpy.linalg.norm(station_loc - sat_loc)
    return (
        2 * mu / tec.C**2 * numpy.log(
            (station_r + sat_r + distance_r)
            / (station_r + sat_r - distance_r)
        )
    )

def clock_cor(meas):
    return -2 * np.inner(meas.sat_pos, meas.sat_vel) / tec.C**2



def estimate_Bc(meas):
#    meas = station_data[sta][prn][tick]
    phase, pseudorange, wavelength = tec.ionosphere_free(meas)
    return phase - pseudorange, wavelength

def bias(signal):
    def f(meas):
        res = signal(meas)
        return res[0] - res[1], res[-1]
    return f

def geometry_free_solve(ddn1, ddn2, ws, station_data, sta1, sta2, prn1, prn2, ticks):
    lambda_1 = lambda_1s[prn1[0]]
    lambda_2 = lambda_2s[prn1[0]]

    # Φ_i - R_i = B_i + err  with B_i = b_i + λ_1*N_1 - λ_2*N_2
    B_i = bias(tec.geometry_free)
    
    Bis = [0, 0, 0, 0]

    for i, (sta, prn) in enumerate(product([sta1, sta2], [prn1, prn2])):
        B_i_samples = []
        for tick in ticks:
            B_i_samples.append( B_i(station_data[sta][prn][tick])[0] )
        print(numpy.mean(B_i_samples), numpy.std(B_i_samples))
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
    print(n1s, n2s, err)
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
    return n1s, n2s, err

def geometry_free_solve_(ddn1, ddn2, ws, station_data, sta1, sta2, prn1, prn2, ticks):
    lambda_1 = lambda_1s[prn1[0]]
    lambda_2 = lambda_2s[prn1[0]]

    # Φ_i - R_i = B_i + err  with B_i = b_i + λ_1*N_1 - λ_2*N_2
    B_i = bias(tec.geometry_free)
    
    sol = Optimize()
#    sol = Solver()
    errs = Reals('err_11 err_12 err_21 err_22')
    n1s = Ints('n1_11 n1_12 n1_21 n1_22')
    n2s = Ints('n2_11 n2_12 n2_21 n2_22')

    sol.add(n1s[0] - n1s[1] - n1s[2] + n1s[3] == ddn1)
    sol.add(n2s[0] - n2s[1] - n2s[2] + n2s[3] == ddn2)

    for i, (sta, prn) in enumerate(product([sta1, sta2], [prn1, prn2])):
        sol.add(n1s[i] - n2s[i] == ws[i])
        B_i_samples = []
        for tick in ticks:
            B_i_samples.append( B_i(station_data[sta][prn][tick])[0] )
        B_i_avg = numpy.mean(B_i_samples)
#        B_i_avg = B_i_samples[0]
        print(B_i_avg, numpy.std(B_i_samples))
        sol.add(lambda_1 * ToReal(n1s[i]) - lambda_2 * ToReal(n2s[i]) + errs[i] > B_i_avg)
        sol.add(lambda_1 * ToReal(n1s[i]) - lambda_2 * ToReal(n2s[i]) - errs[i] < B_i_avg)
    """
        sol.add(errs[0] < .9)
        sol.add(errs[1] < .9)
        sol.add(errs[2] < .9)
        sol.add(errs[3] < .9)
    """
    #sol.add(errs[0] + errs[1] + errs[2] + errs[3] < 17)
    objective = sol.minimize(errs[0] + errs[1] + errs[2] + errs[3])
    if sol.check() != sat:
        return None
    sol.lower(objective)
    if sol.check() != sat:
        return None
#    sol.add(errs[0] + errs[1] + errs[2] + errs[3] < 2)
    # can't do L2 norm with z3, L1 will have to do...
#    sol.(errs[0] + errs[1] + errs[2] + errs[3])

    
    return (
        [sol.model()[n1s[i]].as_long() for i in range(4)],
        [sol.model()[n2s[i]].as_long() for i in range(4)],
        [frac_to_float(sol.model()[errs[i]]) for i in range(4)],
    )

def test_n1_(N_w, station_data, sta, prn, ticks):
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

    # TODO don't average out... look at results for each tick?
    B_c = numpy.mean([
        estimate_Bc(station_data[sta][prn][tick])[0] for tick in ticks
    ])
    gf_bias = bias(tec.geometry_free)
    B_i = numpy.mean([
        gf_bias(station_data[sta][prn][tick]) for tick in ticks
    ])


    N_1_best = None
    err_best = 10000

    for N_1_cand in range(-400, 400):
        N_2_cand = N_1_cand - N_w

        # TODO ???
        N_1_cand, N_2_cand = N_2_cand, N_1_cand

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



def solve_ambiguities(station_data, sta1, sta2, prn1, prn2, tick0, tickn):
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

    # step 0/1: remove ticks with bad data + get widelane dd (∆∇N_W)
    widelane_dds = []
    ticks = []

    for tick in range(tick0, tickn):
        w = widelane_ambiguity(station_data, sta1, sta2, prn1, prn2, tick)
        if math.isnan(w):
            continue
        widelane_dds.append(w)
        ticks.append(tick)

    widelane_dd = numpy.mean(widelane_dds)
    print("wideland double difference: {0:0.3f} +/- {1:0.4f}".format(
        widelane_dd, numpy.std(widelane_dds)
    ))

    widelane_dd = round(widelane_dd)

    # step 2:
    # i) estimate ∆∇Bc
    ddbcs = [
        double_difference(
            bias(tec.ionosphere_free),
            station_data, sta1, sta2, prn1, prn2, tick
        ) for tick in ticks
    ]
    ddbc = numpy.mean(ddbcs)
    print("ionosphere free bias dd: {0:0.3f} +/- {1:0.4f}".format(
        ddbc, numpy.std(ddbcs)
    ))

    # ii) estimate ∆∇N_1
    ddn1 = (ddbc / lambda_n) - (lambda_w * widelane_dd / lambda_2)
    print("n1 double difference: {0:0.3f}".format(ddn1))
    ddn1 = round(ddn1)

    # ii) estimate ∆∇N_2 (N_W = N_1 - N_2)
    ddn2 = ddn1 - widelane_dd
    print("n2 double difference: {0:0.3f}".format(ddn2))

    # TODO somehow ddn1 and ddn2 are swapped???
    #ddn1, ddn2 = ddn2, ddn1

    # step 3:
    # estimate N_Ws
    ws, errs, _ = widelane_solve(widelane_dd, station_data, sta1, sta2, prn1, prn2, ticks[0])
    print(ws)
    print(errs)

    return ( geometry_free_solve(ddn1, ddn2, ws, station_data, sta1, sta2, prn1, prn2, ticks) )


def solve_ambiguities_(station_data, sta1, sta2, prn1, prn2, tick0, tickn):
    # step 0: remove ticks where we don't have all the data we need...
    # step 1: find widelane double difference to estimate ∆∇N_W
    # step 2: solve N_Ws
    # step 3: use N_Ws to solve for probable N_1s

    # initialize wavelengths for this frequency band
    lambda_1 = lambda_1s[prn1[0]]
    lambda_2 = lambda_2s[prn1[0]]
    lambda_n = lambda_ns[prn1[0]]
    lambda_w = lambda_ws[prn1[0]]

    # step 0/1: remove ticks with bad data + get widelane dd (∆∇N_W)
    widelane_dds = []
    ticks = []

    for tick in range(tick0, tickn):
        w = widelane_ambiguity(station_data, sta1, sta2, prn1, prn2, tick)
        if math.isnan(w):
            continue
        widelane_dds.append(w)
        ticks.append(tick)

    widelane_dd = numpy.mean(widelane_dds)
    print("wideland double difference: {0:0.3f} +/- {1:0.4f}".format(
        widelane_dd, numpy.std(widelane_dds)
    ))

    widelane_dd = round(widelane_dd)

    # step 2:
    # estimate N_Ws
    ws, errs, _ = widelane_solve(widelane_dd, station_data, sta1, sta2, prn1, prn2, ticks[0])

    # step 3:
    ns = []
    for i, (sta, prn) in enumerate(product([sta1, sta2], [prn1, prn2])):
        n1, err = test_n1(ws[i], station_data, sta, prn, ticks)
        n2 = n1 - ws[i]
        ns.append( (n1, n2, err) )
    
    return ns

def lambda_parms(y, A, B, Qy):
    Qyinv = numpy.linalg.inv(Qy)
    print(Qyinv)

    Pb = B @ numpy.linalg.inv(B.T @ Qyinv @ B) @ B.T @ Qyinv
    Pb_perp = numpy.eye(2) - Pb
    Abar = Pb_perp @ A

    print(Abar)

    print(Abar.T @ Qyinv @ Abar)
    Qa = numpy.linalg.inv(Abar.T @ Qyinv @ Abar)

    ahat = Qa @ Abar.T @ Qyinv @ y
    return ahat, Qa
