"""
Tests for bias solving routines
"""
from dataclasses import dataclass
import random
from tid import bias_solve
from typing import cast, Any, Dict, List

import numpy

from laika.lib import coordinates


@dataclass
class ConnTickMap:
    connections: List


@dataclass
class FakeConnection:
    station: str
    prn: str
    ticks: numpy.ndarray
    ipps: numpy.ndarray
    vtecs: numpy.ndarray
    is_glonass: bool
    glonass_chan: int


@dataclass
class FakeScenario:
    conn_map: Dict


def generate_data(station_count=4, sat_count=4, duration=240):
    """
    Args:
        station_count: number of fake stations
        sat_count: number of fake satellites
        duration: total tick number

    Returns:
        FakseScenario
    """
    stations = [f"station_{i}" for i in range(station_count)]
    sats = [f"sat_{i}" for i in range(sat_count)]
    sat_glonass = {sat: random.choice([None, -3, 2, 6]) for sat in sats}

    station_biases = dict(
        zip(stations, (numpy.random.rand(station_count, 3) - 0.5) * 20)
    )
    sat_biases = dict(zip(sats, (numpy.random.rand(sat_count) - 0.5) * 20))

    # populate 16 places where ipps can occur, measurements will randomly
    # be spread out among them
    lats, lons = numpy.meshgrid(numpy.arange(20, 40, 5), numpy.arange(20, 40, 5))
    coords = numpy.stack(
        (numpy.reshape(lats, 16), numpy.reshape(lons, 16), numpy.zeros(16)), axis=1
    )

    # ipps from connections are ECEF XYZ
    ipps_allowed = coordinates.geodetic2ecef(coords)

    # pick TECu values for each IPP and each tick
    TEC_truths = numpy.random.rand(duration, 16) * 50

    connection_data = cast(Dict[str, Dict[str, Any]], {})
    for station in stations:
        connection_data[station] = dict()
        for sat in sats:
            ipps = numpy.zeros((duration, 3))
            vtecs = numpy.zeros((duration, 2))
            ticks = numpy.arange(duration)

            is_glonass = sat_glonass[sat] is not None
            glonass_chan = sat_glonass[sat] or 0

            for tick in ticks:
                # pick a location
                loc = random.randint(0, 15)
                # pick a slant
                slant = numpy.random.rand() * 0.5 + 0.25

                TEC = TEC_truths[tick][loc]
                ipps[tick] = ipps_allowed[loc]

                if not is_glonass:
                    station_bias = station_biases[station][0]
                else:
                    station_bias = (
                        station_biases[station][1]
                        + station_biases[station][2] * glonass_chan
                    )
                measurement = TEC + (station_bias - sat_biases[sat]) * slant
                vtecs[tick] = (measurement, slant)

            connection_data[station][sat] = ConnTickMap(
                [
                    FakeConnection(
                        station, sat, ticks, ipps, vtecs.T, is_glonass, glonass_chan
                    )
                ]
            )

    return FakeScenario(connection_data), sat_biases, station_biases


def true_error(
    true_sat_biases, true_station_biases, calcd_sat_biases, calcd_station_biases
):
    """
    How far off were the biases from the true biases?
    """

    sum_squared_error = 0.0
    for station in true_station_biases:
        sum_squared_error += cast(
            float,
            numpy.linalg.norm(
                true_station_biases[station] - calcd_station_biases[station]
            ),
        )

    for sat in true_sat_biases:
        sum_squared_error += (true_sat_biases[sat] - calcd_sat_biases[sat]) ** 2

    return sum_squared_error


def bias_solve_error(station_count=10, sat_count=10, duration=240):
    """
    Args:
        station_count: number of fake stations
        sat_count: number of fake satellites
        duration: total tick number

    Returns:
        mean squared error in all bias calculations
    """
    fake_sc, sat_biases, station_biases = generate_data(
        station_count, sat_count, duration
    )
    bsolver = bias_solve.SimpleBiasSolver(fake_sc)
    calcd_sat_biases, calcd_station_biases = bsolver.solve_biases()

    return true_error(
        sat_biases, station_biases, calcd_sat_biases, calcd_station_biases
    ) / (len(sat_biases) + len(station_biases) * 3)


def test_bias_solver():
    """
    Run bias solver 200 times with simple simulations. Verify that the mean
    squared error for parameters is <3.0 50% of the time, and <6.0 90% of the time
    """
    errors = [bias_solve_error(duration=1) for _ in range(200)]
    assert numpy.quantile(errors, 0.5) < 3.0
    assert numpy.quantile(errors, 0.9) < 6.0
