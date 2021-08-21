
# get estimate of total electron content for a given location for a time range

# TEC = (p_f1 - p_f2) / (k * (1 / f1**2 - 1 / f2**2))

import numpy
import math

from laika import constants, helpers, gps_time
from laika.lib import coordinates

K = 40.308e16
m_to_TEC = 6.158  # meters of L1 error to TEC
# set ionosphere puncture to 350km
IONOSPHERE_H = constants.EARTH_RADIUS + 350000
# maximum density of electrons for slant calculation
IONOSPHERE_MAX_D = constants.EARTH_RADIUS + 350000
C = constants.SPEED_OF_LIGHT

# TODO: glonass if FDMA, so this doesn't work :/
#  and galileo is more complicated and needs l5a and l5b
#  so ignore this for anything but GPS...
F_lookup = {
    'R':(constants.GLONASS_L1, constants.GLONASS_L2, constants.GLONASS_L5),
    'G':(constants.GPS_L1, constants.GPS_L2, constants.GPS_L5),
    'E':(constants.l1, constants.l5, math.nan)
}

def gps_time_from_dense_record(np_meas):
    '''
    A little convenience function to get a laika.GPSTime object from the values in one of the GNSS
    measurements. When we were using the laika.GNSSMeasurement data structure this wouldn't have been
    necessary, but now that everything is crammed into the structured-numpy array all that is in there
    is the week & second (in GPS-time), so it's necessary sometimes to turn that back into the laika
    GPSTime object.

    Parameters
    ----------
    np_meas : one row of a numpy structured array, as in the Scenario._station_data object.

    Returns : laika.GPSTime
    -------

    '''
    myrecvtime = gps_time.GPSTime(week=np_meas['recv_time_week'][0], tow=np_meas['recv_time_sec'][0])
    return myrecvtime

def correct_tec(tec_entry, rcvr_bias=0, sat_bias=0):
    # NOTE: the biases now have units of TECu, not meters !
    location, vtecish, s_to_v = tec_entry
    stec = vtecish / s_to_v  + (sat_bias - rcvr_bias)
    return location, stec * s_to_v, s_to_v

def correct_tec_vals(vtecish, s_to_v, rcvr_bias=0., sat_bias=0.):
    # Same as function 'correct_tec' but with scalar input rather than tuple
    return (vtecish / s_to_v + (sat_bias - rcvr_bias)) * s_to_v

def calc_vtec(scenario, station, prn, tick, ionh=IONOSPHERE_H, el_cut=0.3, n1=0, n2=0, offset=0, rcvr_bias=0, sat_bias=0):
    """
    Given a receiver position and measurement from a specific sv this calculates the vertical TEC, and the point
    at which that TEC applies (between the sat and the rec). Checks to see whether the satellite is too low in
    the sky and returns None if so.

    Here the STEC estimate is pretty simple:

        STEC = [f_1**2 X f_2**2/(f_1**2 - f_2**2)] X [(Phi_1 - n1)/f_1 - (Phi_2 - n2)/f_2]*C
                |----  ^Delay Factor      -----|     |-----   Carrier-Phase Delay    ------|

    If we have receiver or satellite clock biases, those are included.
    Returns: ( VTEC (i.e. STEC * Slant_to_Vertical , Ionosphere-piercing loc , S-to-V factor )
    """
    datastruct = scenario.station_data_structure
    meas = scenario.get_measure(station, prn, tick) #generic, will return either data type
    if meas is None:
        return None

    rec_pos = scenario.station_locs[station]

    n1 = n1 or 0
    n2 = n2 or 0
    offset = offset or 0

    # Calculate the slant TEC.
    if datastruct=='dense':
        res = calc_carrier_delay_dense(
            scenario.dog,
            meas,
            n1=n1, n2=n2, offset=offset,
            rcvr_bias=rcvr_bias,
            sat_bias=sat_bias,
        ) # --> res = (phase_diff_meters, delay_factor)
        my_sat_pos = numpy.array([meas['sat_pos_x'][0], meas['sat_pos_y'][0] , meas['sat_pos_z'][0]])
    else:
        res = calc_carrier_delay(
            scenario.dog,
            meas,
            n1=n1, n2=n2, offset=offset,
            rcvr_bias=rcvr_bias,
            sat_bias=sat_bias,
        )
        my_sat_pos = meas.sat_pos

    if res is None:
        return None

    phase_diff, delay_factor = res
    stec = phase_diff * delay_factor / K

    if math.isnan(stec):
        return None

    # -- Commenting the next lines out because data imoprt step does all processing --
    # if not measurement.processed:
    #     measurement.process(scenario.dog)
    el = scenario.station_el(station, my_sat_pos)
    # ignore things too low in the sky
    if el < el_cut or math.isnan(el):
        return None
    s_to_v = s_to_v_factor(el, ionh=ionh)
    return stec * s_to_v, ion_loc(rec_pos, my_sat_pos), s_to_v

def s_to_v_factor(el, ionh=IONOSPHERE_H):
    '''
    Converts between S and V. (Slant and Vertical?). See the Springer Handbook, equation (6.99) for this
    formula. Note that this is raised to the 1/2 power instead of -1/2 because we are going from STEC
    to VTEC but (6.99) goes the other way.
    :param el:  Elevation angle
    :param ionh: Height of the ionosphere in the zenith direction. Note that this shoould
                    already have Earth's Radius added in (as IONOSPHERE_H does above).
    :return:
    '''
    return math.sqrt(1 - (math.cos(el) * constants.EARTH_RADIUS / ionh) ** 2)

def ion_loc(rec_pos, sat_pos):
    """
    Given a receiver position and sat position, figure out
    where the ionosphere boundary between the two lines
    http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    """
    a = sum( (sat_pos - rec_pos)**2 )
    b = 2 * sum(( sat_pos - rec_pos) * rec_pos )
    c = sum(rec_pos**2) - IONOSPHERE_H**2

    t0 = (-b + math.sqrt(b ** 2 - (4 * a * c))) / (2 * a)
    t1 = (-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a)

    res0 = numpy.array([(rec_pos[d] + (sat_pos[d] - rec_pos[d]) * t0) for d in range(3)])
    res1 = numpy.array([(rec_pos[d] + (sat_pos[d] - rec_pos[d]) * t1) for d in range(3)])

    # one of those intersections is the other side of the world...
    if (sum((rec_pos - res1)**2)) > (sum((rec_pos - res0)**2)):
        return res0
    else:
        return res1

def get_band_freq_for_dense_measurement(dog, np_meas):
    '''Takes care of some of the boilerplate in the other functions.'''
    if np_meas is None:
        return None, None, None

    my_meas_prn = np_meas['prn'][0]
    band_1 = 'L1C'
    if my_meas_prn[0] == 'E':
        band_2 = 'L5C'
    else:
        band_2 = 'L2C'

    two_bands = [band_1, band_2]
    # if my_meas_prn[0] == 'R':
    my_recv_time = gps_time.GPSTime(week=np_meas['recv_time_week'], tow=np_meas['recv_time_sec'])
    freqs = [dog.get_frequency(my_meas_prn, my_recv_time, band) for band in two_bands]

    if (math.isnan(np_meas[band_1][0]) or math.isnan(np_meas[band_2][0])):
        return None, None, None
    return freqs, two_bands, my_recv_time


def calc_carrier_delay(dog, measurement, n1=0, n2=0, offset=0, rcvr_bias=0, sat_bias=0):
    """
    calculates the carrier phase delay associated with a measurement
    """
    if measurement is None:
        return None
    observable = measurement.observables

    band_1 = 'L1C'
    if measurement.prn[0] == 'E':
        band_2 = 'L5C'
    else:
        band_2 = 'L2C'

    freqs = [dog.get_frequency(measurement.prn, measurement.recv_time, band) for band in [band_1, band_2]]

    if (
        math.isnan(observable.get(band_1, math.nan))
        or math.isnan(observable.get(band_2, math.nan))
    ):
        # missing info: can't do the calculation
        return None

    delay_factor = freqs[0]**2 * freqs[1]**2/(freqs[0]**2 - freqs[1]**2)

    phase_diff_meters = C * (
        (observable[band_1] - n1)/freqs[0] - (observable[band_2] - n2)/freqs[1]
    )
    return phase_diff_meters + offset + sat_bias - rcvr_bias, delay_factor

def calc_carrier_delay_dense(dog, np_meas, n1=0., n2=0., offset=0., rcvr_bias=0, sat_bias=0):
    """
    calculates the carrier phase delay associated with a measurement, given in the numpy
    structured array format.

    Note: if both n1/n2 and offset are provided, offset will be IGNORED!

    returns (phase_diff_meters, delay_factor)
    """
    if (n1 != 0. or n2 != 0.) and (offset != 0.):
        # print('WARNING: both n1/n2 and offset were provided to function `calc_carrier_delay_dense(..)`. Offset will be ignored.')
        offset = 0.
    n1 = n1 if n1 is not None else 0.
    n2 = n2 if n2 is not None else 0.
    offset = offset if offset is not None else 0.

    freqs, bands, my_recv_time = get_band_freq_for_dense_measurement(dog, np_meas)
    if freqs is None and bands is None and my_recv_time is None:
        return None
    band_1 = bands[0]; band_2 = bands[1];

    delay_factor = freqs[0]**2 * freqs[1]**2/(freqs[0]**2 - freqs[1]**2)

    phase_diff_meters = C * (
        (np_meas[band_1][0] - n1)/freqs[0] - (np_meas[band_2][0] - n2)/freqs[1]
    )
    return phase_diff_meters + offset + sat_bias - rcvr_bias, delay_factor


def melbourne_wubbena(dog, measurement):
    """
    Melbourne-Wubbena Combination:  (Carrier-Phase Widelane - Code-Phase Narrowlane)
                                    (\phi_WL - R_NL)
    Wide lane measurement, should not change "much" for GPS, except
    due to cycle slips.
    GLONASS is different though... and this doesn't work so well
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'

    freqs = [dog.get_frequency(measurement.prn, measurement.recv_time, band) for band in ['C1C', 'C2C']]
    phase = C/(freqs[0] - freqs[1])*(observable['L1C'] - observable['L2C'])
    pseudorange = 1/(freqs[0] + freqs[1])*(freqs[0]*observable['C1C'] + freqs[1]*observable[chan2])
    wavelength = C/(freqs[0] - freqs[1])
    return phase - pseudorange, wavelength

def melbourne_wubbena_dense(dog, np_meas):
    """
    Same as function above but for the measurement in the numpy structured array format.
    """
    if np_meas is None:
        return None

    chan2 = 'C2C' if not numpy.isnan(np_meas['C2C'][0]) else 'C2P'
    recv_time = gps_time.GPSTime(week=np_meas['recv_time_week'][0], tow=np_meas['recv_time_sec'][0])
    # freqs = [dog.get_frequency(np_meas['prn'][0], recv_time, band) for band in ['C1C', 'C2C']]
    freqs = (constants.GPS_L1, constants.GPS_L2) if np_meas['prn'][0][0]=='G' else [dog.get_frequency(np_meas['prn'][0], recv_time, band) for band in ['C1C', 'C2C']]
    phase = C/(freqs[0] - freqs[1])*(np_meas['L1C'][0] - np_meas['L2C'][0])
    pseudorange = 1/(freqs[0] + freqs[1])*(freqs[0]*np_meas['C1C'][0] + freqs[1]*np_meas[chan2][0])
    wavelength = C/(freqs[0] - freqs[1])
    return phase - pseudorange, wavelength

def melbourne_wubbena_vector(np_meas, ticks, ticks_to_rows, np_chan2_vec=None):
    '''
    function to efficiently compute the Melbourne-Wubbena combination over a vector of many ticks.

    Note: Currently this will only work for GPS, so may need to be generalized in the future.

    Parameters
    ----------
    np_meas : np.ndarray
        Structured numpy array for this station (right from the scenario ._station_data object)
    ticks : list
        list of ticks involved in this conneciton object
    ticks_to_rows : dict
        dict object that maps a tick to a row of the 'np_meas' matrix
    np_chan2_vec : None
        (not implemented)

    Returns : np.ndarray
        A vector containing the melbourne-wubenna combination of the obserables over these ticks.
    -------

    '''
    F1=constants.GPS_L1; F2=constants.GPS_L2;
    assert np_meas['prn'][ticks_to_rows[ticks[0]]][0][0]=='G', "Function only works for GPS right now."
    # Combine 'C2P' with 'C2C' where it is missing:
    if np_chan2_vec is None:
        np_chan2_vec = numpy.where(numpy.isnan(np_meas['C2C']), np_meas['C2P'], np_meas['C2C'])
    trows = list(map(lambda x: ticks_to_rows[x], ticks)) # row indices matching each tick
    MWvec = C/(F1-F2)*(np_meas['L1C'][trows,0] - np_meas['L2C'][trows,0]) - \
            1/(F1+F2)*(F1*np_meas['C1C'][trows,0] + F2*np_chan2_vec[trows,0])
    return MWvec

def melbourne_wubbena_vector_from_conn(scen, conn):
    '''
    Another helper function to make the MW vector directly from the scenario/connection objects.

    Parameters
    ----------
    scen : <pytid.get_data.ScenarioInfoDense> object
        Scenario from which this connection was taken
    conn : <pytid.connections.Connection> object
        Connection object
    Returns : np.ndarray
        Melbourne-Wubenna vector on observables over the course of the connection
    -------

    '''
    # F1 = constants.GPS_L1; F2 = constants.GPS_L2;
    return melbourne_wubbena_vector(scen._station_data[conn.station], conn.ticks,
                                    scen.row_by_prn_tick_index[conn.station][conn.prn])

def wide_lane(measurement):
    """
    For Carrier-Phase (cycles): \phi_WL = (\phi_1 * f_1 - \phi_2 * f_2) / (f_1 - f_2)
           Code-Phase:
    Wide lane measurement: acts as measurement with wider wavelength
        used for ambiguity correction
    Note: This function returns carrier-phase Widelane denominated in meters, not cycles
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    phase = C/(freqs[0] - freqs[1])*(observable['L1C'] - observable['L2C'])
    pseudorange = 1/(freqs[0] - freqs[1])*(freqs[0]*observable['C1C'] - freqs[1]*observable[chan2])
    wavelength = C/(freqs[0] - freqs[1])
    return phase, pseudorange, wavelength

def narrow_lane(measurement):
    """
    Narrow lane measurement: acts as measurement with narrower wavelength
    used for ambiguity correction
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    phase = C/(freqs[0] + freqs[1])*(observable['L1C'] + observable['L2C'])
    pseudorange = 1/(freqs[0] + freqs[1])*(freqs[0]*observable['C1C'] + freqs[1]*observable[chan2])
    wavelength = C/(freqs[0] + freqs[1])
    return phase, pseudorange, wavelength

def ionosphere_free(measurement):
    """
    ionosphere free signal combination
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    phase = C/(freqs[0]**2 + freqs[1]**2)*(observable['L1C']*freqs[0] + observable['L2C']*freqs[1])
    pseudorange = 1/(freqs[0]**2 + freqs[1]**2)*(freqs[0]**2*observable['C1C'] + freqs[1]**2*observable[chan2])
    wavelength = C/(freqs[0] + freqs[1])  # same as narrow lane combination
    return phase, pseudorange, wavelength

def geometry_free(measurement):
    """
    geometry free (ionosphere) signal combination
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    phase = C*(observable['L1C']/freqs[0] - observable['L2C']/freqs[1])
    # yes, pseudorange is flipped intentionally
    pseudorange = observable[chan2] - observable['C1C']
    return phase, pseudorange

def ionosphere_combination(measurement, n1=0, n2=0):
    """
    combining the narrowland and widelane combinations to get
    a signal where the ionosphere is counted twice
    eq 8.28 in Petrovski and Tsujii Digital Satellite Navigation and Geophysics
    kind of like an anti-Melbourne Wubbena
    """

    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    N_w = n1 - n2
    N_n = n1 + n2

    lambda_w = C/(freqs[0] - freqs[1])
    phi_w = lambda_w * (observable['L1C'] - observable['L2C'])

    lambda_n = C/(freqs[0] + freqs[1])
    phi_n = lambda_n * (observable['L1C'] + observable['L2C'])

    # phi_w - phi_n = 2*I - lambda_n * N_n + lambda_w * N_w + (errors)
    ion = phi_w - phi_n + lambda_n * N_n - lambda_w * N_w

    return ion/2