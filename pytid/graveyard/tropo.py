"""
tropospheric modeling...
ESA_GNSS-Book_TM-23_Vol_Ip section 5.4.2
"""
from laika.lib import coordinates
import numpy
import math


# ESA GNSS VolI tbl 5.2 + 5.3 : Coefficients of the wet mapping function
# converted to radians...
mapping_lats = [0.261799388, 0.52359878, 0.78539816, 1.04719755, 1.308996939]

mdavg = {
    'a': [1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3],
    'b': [2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3],
    'c': [62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3]
}
mdamp = {
    'a': [0.e0,1.2709626e-5,2.6523662e-5,3.4000452e-5,4.1202191e-5],
    'b': [0.e0,2.1414979e-5,3.0160779e-5,7.2562722e-5,11.723375e-5],
    'c': [0.e0,9.0128400e-5,4.3497037e-5,84.795348e-5,170.37206e-5]
}
height_cor = [2.53e-5, 5.49e-3, 1.14e-3]

mws = {
    'a': [7e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4],
    'b': [1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3],
    'c': [4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2]
}

# constants from ESA GNSS VolI eqn 5.63 + 5.64
k1 = 77.604  # units of K/mbar
k2 = 382000  # units of K**2/mbar
R_d = 287.054  # units of J/(kg * K)
g_m = 9.784  # units of m/s**2
g = 9.80665  # units of m/s**2

# ESA GNSS VolI tbl 5.1
metavgs = {
    'P0': [1013.25, 1017.25, 1015.75,1011.75, 1013.00],
    'T0': [299.65, 294.15, 283.15, 272.15, 263.65],
    'e0': [26.31, 21.79, 11.66, 6.78, 4.11],
    'B0': [6.30E-3, 6.05E-3, 5.58E-3, 5.39E-3, 4.53E-3],
    'l0': [2.77, 3.15, 2.57, 1.81, 1.55]
}
metamps = {
    'P0': [0.0, -3.75,  -2.25, -1.75, -0.50],
    'T0': [0.0, 7.0, 11.0, 15.0, 14.5],
    'e0': [0.0, 8.85, 7.24, 5.36, 3.39],
    'B0': [0.0E-3, 0.25E-3, 0.32E-3, 0.81E-3, 0.62E-3],
    'l0': [0.0, 0.33, 0.46, 0.74, 0.30]
}


# neill mapping functions for tropospheric moisture
def mariani_mapping(lat, a, b, c):
    return (
        (1 + a/(1 + b/(1 + c)))
        / (numpy.sin(lat) + a/(numpy.sin(lat) + b/(numpy.sin(lat) + c)))
    )

def M_wet(lat):
    # TODO uh this gives crazy high values at the equator? is that right?
    def wet_param(label):
        return numpy.interp(abs(lat), mapping_lats, mws[label])
    
    return mariani_mapping(
        lat,
        wet_param('a'),
        wet_param('b'),
        wet_param('c'),
    )

def seasonal_variation(gpstime, lat, avgs, amps):
    # northern and southern hemispheres out of phase
    offset = 28 if lat > 0 else 211
    season = gpstime.as_datetime().timetuple().tm_yday - offset
    # convert to radians
    season *= 2 * math.pi / 365.25
    return (
        numpy.interp(abs(lat), mapping_lats, avgs)
        - numpy.interp(abs(lat), mapping_lats, amps) * numpy.cos(season)
    )


def M_dry(lat, height, gpstime):
    def dry_param(label):
        return seasonal_variation(gpstime, lat, mdavg[label], mdamp[label])
    
    return (
        mariani_mapping(
            lat,
            dry_param('a'),
            dry_param('b'),
            dry_param('c'),
        )
        + height * (1/numpy.sin(lat) - mariani_mapping(lat, *height_cor))
    )

def Trz(lat, height, gpstime):
    """
    calculate Tr_z,d, Tr_z,w, Tr_z0,d, Tr_z0,w
    see ESA GNSS VolI eqn 5.63 + 5.64
    """
    def met_param(label):
        return seasonal_variation(gpstime, lat, metavgs[label], metamps[label])
    
    trz0d = 1e-6 * k1 * R_d * met_param('P0') / g_m
    trz0w = (
        1e-6 * k2 * R_d * met_param('e0')
        / ((
            ((met_param('l0') + 1) * g_m)
            - (met_param('B0') * R_d)
        ) * met_param('T0'))
    )
    
    base = (1 - met_param('B0') * height / met_param('T0'))
    trzd_pow = g / (R_d * met_param('B0'))
    trzd = base ** trzd_pow * trz0d

    trzw_pow = ((met_param('l0') + 1) * g) / (R_d * met_param('B0')) - 1
    trzw = base ** trzw_pow * trz0w

    return trzd, trzw, trz0d, trz0w

def tropo_delay(measurement, height=0):
    # base tropospheric delay Tr(E)

    lat = numpy.radians(coordinates.ecef2geodetic(measurement.sat_pos)[0])

    trzd, trzw, _, _ = Trz(lat, height, measurement.recv_time)
    m_d = M_dry(lat, height, measurement.recv_time)
    m_w = M_wet(lat)

    return trzd * m_d + trzw * m_w
