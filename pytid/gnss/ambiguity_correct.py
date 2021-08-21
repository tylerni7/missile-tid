from collections.abc import Iterable
import ctypes
from itertools import product
import functools, datetime
import math, os
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
    '''
    Calculates N1, N2, N3, N4 based on minimizing the SSE (i.e. the
    brute_force_dd routine). Starts by computing the Melb-Wubb for
    every time t:
        B_W(t) = \Phi_W(t) - R_N(t) # for t in ticks
    then takes an average:
        vr1s1 = mean([B_W(t)])
        ...
    '''
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
    '''????'''
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
    '''DEPRECATED AS OF 2/26/21.

    This hasn't been used in a while but Tyler wrote it early on so there may have been something in mind for it.
    At one point it looks like it was called at the end of the solve_ambiguities() function below, but has een
    commented out.'''
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


def rho(station_locs, station_data, station, prn, tick):
    '''
    Returns the Euclidean distance in 3d between the satellite and the receiver. If the laika object has
    been corrected, use that. This is based on the old data structure.
    '''
    if station_data[station][prn][tick].corrected:
        return numpy.linalg.norm(station_locs[station] - station_data[station][prn][tick].sat_pos_final)
    else:
        return numpy.linalg.norm(station_locs[station] - station_data[station][prn][tick].sat_pos)

def rho_factory(scenario, station):
    recv_pos = scenario.station_locs[station]
    def my_rho(my_sat_pos):
        return numpy.linalg.norm(recv_pos - my_sat_pos)
    return my_rho

def offset(scenario, sta, prn, ticks):
    """
    Sidestep ambiguity correction by just finding the difference between the
    absolute but noisy code phase data and the smooth but offset carrier phase
    data.

    Returns the offset to be applied to the L1-L2 value and its error
    Note: Assumes no cycle slips
    """
    datastruct = scenario.station_data_structure
    if datastruct=='dense':
        prn_rows = numpy.where(scenario.station_data[sta]['prn']==prn)[0]
        prn_ticks = scenario.station_data[sta]['tick'][prn_rows][:,0]
        tick_to_row = dict(zip(prn_ticks, prn_rows))
        tick0_row = numpy.where((scenario.station_data[sta]['prn']==prn) & (scenario.station_data[sta]['tick']==ticks[0]))[0][0]
        my_recv_time = tec.gps_time_from_dense_record(scenario.station_data[sta][tick0_row])
    else:
        my_recv_time = scenario.station_data[sta][prn][ticks[0]].recv_time

    freq_1, freq_2 = [
        scenario.dog.get_frequency(
            # prn, scenario.station_data[sta][prn][ticks[0]].recv_time, band
            prn, my_recv_time, band
        )
        for band in ["C1C", "C2C"]
    ]

    if datastruct=='dense':
        def obs(obs_tick, chan):
            res = scenario.station_data[sta][chan][tick_to_row[obs_tick]][0]
            if math.isnan(res) and chan == 'C2C':
                res = scenario.station_data[sta][tick_to_row[obs_tick]]['C2P'][0]
            return res
    else:
        def obs(tick, chan):
            res = scenario.station_data[sta][prn][tick].observables.get(chan, math.nan)
            if math.isnan(res) and chan == 'C2C':
                res = scenario.station_data[sta][prn][tick].observables.get('C2P', math.nan)
            return res

    # *** Note that the step where we subtract (sat_clock_err * C) is not needed here because
    #       it will cancel out in the subtraction ***
    c2 = numpy.array([obs(i, 'C2C') for i in ticks])
    c1 = numpy.array([obs(i, 'C1C') for i in ticks])
    l2 = numpy.array([obs(i, 'L2C') for i in ticks])
    l1 = numpy.array([obs(i, 'L1C') for i in ticks])

    # note this is chan2 - chan1 because the ionosphere acts opposite for code phase
    code_phase_diffs = c2 - c1
    carrier_phase_diffs = tec.C * (l1 / freq_1 - l2 / freq_2)

    avg_diff = numpy.mean(code_phase_diffs - carrier_phase_diffs)
    med_diff = numpy.median(code_phase_diffs - carrier_phase_diffs)
    err = numpy.std(code_phase_diffs - carrier_phase_diffs)
    return avg_diff, med_diff, err, numpy.mean(code_phase_diffs), numpy.mean(carrier_phase_diffs)

def solve_ambiguity_least_squares_dense(scenario, sta, prn, ticks, variance_matrix_identity=False,
                                        write_intermediates_to_path=None):
    '''
    This function will calculate the ambiguities N_1 and N_2 for a single connection
    between a satellite and a receiver. It will use raw data (i.e. NOT differenced). This
    is a version specifically for testing modifications, so there will proably be several
    copies of this function and code duplication.

    Currently in this function:
        - N_2 left in the design matrix, equals 1 on L1C, L2C.
        - N_12 estimated from the mean widelane over the period.
        - N_12 subtracted from y for 'L1C' only. I.e. N_1 = N_2(est) + N_12(est) = N_2 + N_1 - N_2
        - There *is* an intercept column, β_0, (column 0)
        - We estimate a variance matrix with Qyy, and Qyy^{-1} with Wyy.

    The setup and example for this implementation is from:
        [Xu2] Xu,Xu "GPS: Theory, Algorithms and Applications", Third Edition. Springer, 2016.
        Section 6.7.1, page 161.
    The solver setup and solutions come from:
        [TM]  Teunisson, Montenbruck (editors) "Springer Handbook of Global Navigation Satellite
              Systems", Springer 2017.
        Section 23.1, page 662.

    Additional Arguments:
        variance_matrix_identity:   If True, uses an identity variance matrix rather than attempting
                                    to differ it for code vs. carrier residuals.
        write_intermediates_to_path:    If given, should be a string containing a path to a folder that
                                        the algorithm can dump a bunch of files to.
    '''
    # initialize wavelengths for this frequency band
    F1, F2, F5 = tec.F_lookup[prn[0]]
    C = tec.C
    lam1 = C/F1; lam2 = C/F2; lam5 = C/F5;

    # Gather \rho, \phi and R_j values
    tick_rows = numpy.array(list(map(lambda x: scenario.row_by_prn_tick_index[sta][prn][x], ticks)))
    n_obs = tick_rows.shape[0]
    sat_errs = scenario.station_data[sta]['sat_clock_err'][tick_rows]

    # TODO: Figure out what the deal was with the sat_clock_err in the laika correct() method...
    phi1_t = scenario.station_data[sta]['L1C'][tick_rows]
    phi2_t = scenario.station_data[sta]['L2C'][tick_rows]
    R1_t = scenario.station_data[sta]['C1C'][tick_rows] #+ sat_errs * C
    R2_t = numpy.where(numpy.logical_not(numpy.isnan(scenario.station_data[sta]['C2C'][tick_rows])),
                       scenario.station_data[sta]['C2C'][tick_rows],
                       scenario.station_data[sta]['C2P'][tick_rows]) #+ sat_errs * C

    # Compute Distances to Satellite: (\rho vector)
    rec_pos = scenario.station_locs[sta]
    sat_pos_mat = numpy.hstack(( scenario.station_data[sta]['sat_pos_x'][tick_rows],
                                 scenario.station_data[sta]['sat_pos_y'][tick_rows],
                                 scenario.station_data[sta]['sat_pos_z'][tick_rows]))
    rho_t = numpy.linalg.norm(rec_pos - sat_pos_mat, axis=1).reshape((n_obs,1))

    # *** Older version for the nested-dict station_data structure ***
    # obs_t =  numpy.array([obs(tick) for tick in ticks])   #obs here spits out a tuple for each t
    # rho_t =  numpy.array([ my_rho(x[1]) for x in obs_t]).reshape((obs_t.shape[0], 1))
    # phi1_t = numpy.array([ob[0]['L1C'] for ob in obs_t]) #in cycles
    # phi2_t = numpy.array([ob[0]['L2C'] for ob in obs_t])
    # R1_t  =  numpy.array([ob[0]['C1C'] for ob in obs_t]) #in meters
    # R2_t  =  numpy.array([ob[0]['C2C'] for ob in obs_t]) #obs here has taken care of making this 'C2P' if needed
    # n_obs = phi1_t.shape[0]

    # N_21 = N_1 - N_2, approximating it with the widelane
    n21_t_widelane = (phi1_t - phi2_t) - (F1-F2)/(F1+F2)*(R1_t/lam1 + R2_t/lam2)
    n21est = numpy.mean(n21_t_widelane)

    # Note the variables used earlier...
    singletons = []; names = [];
    if write_intermediates_to_path is not None and os.path.isdir(write_intermediates_to_path):
        singletons = ['n_obs', 'n21est', ]
        names = ['tick_rows', 'sat_errs', 'phi1_t', 'phi2_t', 'R1_t', 'R2_t', 'rec_pos', 'sat_pos_mat', 'rho_t',
                 'n21_t_widelane']
        # Add the rest down below...

    # Set up the equations like eq. 6.134 (p. 161) of the Springer text "GPS: ..." but
    #   the distance ρ has been subtracted from the left side.
    # | R_1/λ_1-ρ/λ_1 |    | 0   0    f_1^{-1}   f_1 | | N_1 |
    # | R_2/λ_2-ρ/λ_2 |    | 0   0    f_2^{-1}   f_2 | | N_2 |
    # | Φ_1-ρ/λ_1     | =  | 1   0   -f_1^{-1}   f_1 | | B_1 |
    # | Φ_2-ρ/λ_2     |    | 0   1   -f_2^{-1}   f_2 | | C_ρ |
    #
    #   ^ Technically that matrix is underdetermined, but it can still be used to solve
    #   the relevant least squares problem.
    #
    # Alternatively, setting it up to use the widelane N_21 estimate from above:
    #   - Options here: 1) add an intercept, 2) leave first column out.
    #       3) etc...
    # | R_1/λ_1-ρ/λ_1   |    | 1   0    f_1^{-1}   f_1 | | β_0 |
    # | R_2/λ_2-ρ/λ_2   |    | 1   0    f_2^{-1}   f_2 | | N_2 |
    # | Φ_1-ρ/λ_1 - N_21| =  | 1   1   -f_1^{-1}   f_1 | | B_1 |
    # | Φ_2-ρ/λ_2       |    | 1   1   -f_2^{-1}   f_2 | | C_ρ |
    #   - N_21 subtracted from L1C, design matrix now (N x 3)
    #   - then ^N_1 = ^N_2 + mean_t(N_21)
    #   - First column could also be an intercept if desired
    #
    # Everything in Y is in cycles, not meters
    y = numpy.empty((4*n_obs,1), dtype=numpy.float64)
    y[0::4] = (R1_t / lam1 - rho_t / lam1).reshape((n_obs, 1))
    y[1::4] = (R2_t / lam2 - rho_t / lam2).reshape((n_obs, 1))
    y[2::4] = (phi1_t - rho_t / lam1).reshape((n_obs, 1))
    y[3::4] = (phi2_t - rho_t / lam2).reshape((n_obs, 1))
    # y_0 = (R1_t / lam1 - rho_t / lam1 ); print('y_0: %s' % str(y_0.shape))
    # y_1 = (R2_t / lam2 - rho_t / lam2 ); print('y_1: %s' % str(y_1.shape))
    # y_2 = (phi1_t - rho_t / lam1); print('y_2: %s' % str(y_2.shape))
    # y_3 = (phi2_t - rho_t / lam2); print('y_3: %s' % str(y_3.shape))

    # Adjust by N_21:
    y[2::4] = y[2::4] - n21est

    # Xmat is the integer *and* float parts (four columns) of the regression design matrix
    #   - y=A*a + B*b ([TM] eq. 23.2) --> X = [ A   B ]
    Xmat = numpy.zeros((4 * n_obs, 3), dtype=numpy.float64)
    # A[2::4,0] = 1 <-- from the first version of the matrix as above.
    # Xmat[:,0] = 1 #Adding an intercept term. <--*no intercept so it will be full-rank*
    Xmat[3::4,0] = 1; Xmat[2::4,0] = 1;

    # Next two cols of X is the float part.
    freq_vec1 = numpy.array([F1, F2, F1, F2], dtype=numpy.float64)
    freq_vec = numpy.array([1 / F1, 1 / F2, -1 / F1, -1 / F2], dtype=numpy.float64)
    Xmat[:, 1] = numpy.tile(freq_vec, n_obs)
    Xmat[:, 2] = numpy.tile(freq_vec1, n_obs)

    # Was doing some futzing with the variance matrix, though IIRC it didn't make a different
    if variance_matrix_identity:
        sigma_C = 1.0  # Carrier-Phase stddev
        sigma_P = 1.0   # Pseudorange stddev
    else:
        sigma_C = 0.01  # Carrier-Phase stddev
        sigma_P = 1.0  # Pseudorange stddev

    # Inverse Variance Matrix:
    Wyy = numpy.diag(numpy.tile(numpy.array([1. / sigma_P ** 2, 1. / sigma_P ** 2, 1. / sigma_C ** 2, 1. / sigma_C ** 2]),
                                n_obs))
    # *** Note (8/20/21): This was incorrect. Y is arranged as [ y_R1, y_R2, y_phi1, y_phi2 ]_{i}^T stacked vertically,
    #   but this makes it the top half is for R1/R2 and the bottom half is phi1/phi2:
    # Wyy = numpy.diag(numpy.hstack((numpy.ones(n_obs * 2, dtype=numpy.float64) * 1. / sigma_P ** 2,
    #                                numpy.ones(n_obs * 2, dtype=numpy.float64) * 1. / sigma_C ** 2)))

    # B = [X' Wyy X]^{-1} [X' Wyy Y]
    beta = numpy.linalg.inv(Xmat.T.dot(Wyy.dot(Xmat))).dot( Xmat.T.dot(Wyy.dot(y)) )
    N2_hat = beta[0]
    N1_hat = beta[0] + n21est

    if write_intermediates_to_path is not None and os.path.isdir(write_intermediates_to_path):
        singletons += ['n_obs', 'n21est', 'sigma_C', 'sigma_P', 'N2_hat', 'N1_hat']
        names += ['y', 'Xmat', 'Wyy', 'beta']
        npargs = tuple([eval(i) for i in names])
        dump_numpy_arrays_to_folder(write_intermediates_to_path, names, *npargs)
        singleton_dict = {k: eval(k) for k in singletons}
        with open(os.path.join(write_intermediates_to_path, 'singletons.json'),'w') as sjs:
            json.dump(singleton_dict, sjs)
        del names, npargs, singletons, singleton_dict

    return round(N1_hat.item()), round(N2_hat.item()), n21est, N2_hat.item()

def dump_numpy_arrays_to_folder(fold, name_list, *args):
    '''debugging method to accept a folder as the first argument, a list of variable names
    as the second argument, and then an arbitrary number of numpy arrays after. Dumps all
    of them to individual files in the folder. '''
    if len(name_list)!=len(*args):
        print("Note, %s names given and %s numpy args..." % (len(name_list), len(*args)))
        n_dumps = min(len(name_list), len(*args))
    if not os.path.isdir(fold):
        print("%s is not a valid folder, exiting.")
        return

    for i in range(n_dumps):
        pa = os.path.join(fold, name_list[i]+'.tsv')
        if not isinstance(*args[i], numpy.ndarray):
            print("np array argument %d is not a valid numpy array, it is a %s" % (i, str(type(*args[i]))))
            continue
        numpy.savetxt(pa, *args[i], delimiter='\t')
        print("Saved %s..." % name_list[i])

    return


#
# Old code saved:
#

    # ...What follows is the LAMBDA half-integer, half-float solution:
    # Go ahead and solve the float version ([TM] eq. 23.8)
    # QyyInv = numpy.linalg.inv(Qyy)
    # BtQB_inv = numpy.linalg.inv(numpy.dot(numpy.dot(B.T, QyyInv), B))
    # Proj_B = numpy.identity(4*n_obs, dtype=numpy.float64) - numpy.dot(numpy.dot(B,BtQB_inv), numpy.dot(B.T, QyyInv))
    # A_bar = numpy.dot(Proj_B, A)
    # Qaa_hat = numpy.linalg.inv(numpy.dot(numpy.dot(A_bar.T, QyyInv), A_bar)) #[TM] eq. 23.9
    # a_hat_numerator = numpy.dot(numpy.dot(A_bar.T, QyyInv), y)
    # a_hat = numpy.dot(Qaa_hat, a_hat_numerator)
    # b_hat_numerator = numpy.dot(numpy.dot(B.T, QyyInv), (y-numpy.dot(A, a_hat)))
    # b_hat = numpy.dot(BtQB_inv, b_hat_numerator)
    # return round(a_hat[0,0]), round(a_hat[1,0])


