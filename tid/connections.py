"""
Connections are periods of continuous lock (and therefore carrier phase offsets)
between satellites and ground stations.
Things to manage those are stored here
"""
from __future__ import annotations  # defer type annotations due to circular stuff
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Tuple, Union

import numpy

from tid import tec, types, util

# deal with circular type definitions for Scenario
if TYPE_CHECKING:
    from tid.scenario import Scenario


class Connection:
    """
    Each time a receiver acquires a lock on a GNSS satellite,
    some random error of an unknown wavelengths are accumulated
    in the phase measurements. A period of continuous lock
    is referred to as a "connection"

    Therefore, each connection needs to be established so that
    we can solve for the unknown wavelength difference.

    The longer the connection, the more data and better job
    we can do. However, if we mess up, we can introduce extra
    noise.
    """

    def __init__(
        self,
        scenario: Scenario,
        station: str,
        prn: str,
        tick_start: int,
        tick_end: int,
    ) -> None:
        """
        Args:
            scenario: scenario to which this connection belongs
            station: station name
            prn: satellite svid
            tick_start: first possible tick
            tick_end: last existing tick
            filter_ticks: whether we should run processing to determine
                how long this connection lasts, possibly truncating it
        """
        self.scenario = scenario

        self.station = station
        self.prn = prn
        self.tick0 = tick_start
        self.tickn = tick_end

        # integer ambiguities, the phase correction information
        # that is the goal of this whole connections stuff
        self.n_chan1 = None
        self.n_chan2 = None

        # "raw" offset: the difference from the code phase tec values
        self.offset = None  # this value has units of Meters
        self.offset_error = None

    @property
    def is_glonass(self) -> bool:
        """
        Is this a GLONASS satellite?

        Returns:
            boolean indicating glonass or not
        """
        return self.prn.startswith("R")

    @cached_property
    def glonass_chan(self) -> int:
        """
        The channel that GLONASS is using.

        Returns:
            the integer channel GLONASS is using, or 0 if it is not using GLONASS
        """
        if not self.is_glonass:
            return 0
        chan = self.scenario.get_glonass_chan(self.prn, self.observations)
        # can't have gotten None, or we'd not have gotten it in our connection
        assert chan is not None
        return chan

    @cached_property
    def frequencies(self) -> Tuple[float, float]:
        """
        The frequencies that correspond to this connection
        """
        frequencies = self.scenario.get_frequencies(self.prn, self.observations)
        assert frequencies is not None, "Unknown frequencies INSIDE connection object"
        return frequencies

    @cached_property
    def channel2(self) -> str:
        """
        The channel2 name "C2C" or "C2P" associated with this connection's data
        """
        chan2 = util.channel2(self.station, self.prn, self.observations)
        assert chan2, "Unknown channel2 data INSIDE connection object"
        return chan2

    @property
    def ticks(self) -> numpy.ndarray:
        """
        Numpy array of ticks from tick0 to tickn (inclusive), for convenience
        """
        return numpy.arange(self.tick0, self.tickn + 1)

    @property
    def observations(self) -> types.DenseDataType:
        """
        Convenience function: returns the numpy arrays for the raw observations
        corresponding to this connection
        """
        # note: don't use self.ticks, `range` vs `slice` is a lot slower
        assert self.scenario.station_data
        return self.scenario.station_data[self.station][self.prn][
            self.tick0 : self.tickn + 1
        ]

    def elevation(
        self, sat_pos: Union[types.ECEF_XYZ, types.ECEF_XYZ_LIST]
    ) -> Union[types.ECEF_XYZ, types.ECEF_XYZ_LIST]:
        """
        Convenience wrapper around scenario.station_el, but specifically
        for the station that this connection uses.

        sat_pos: numpy array of XYZ ECEF satellite positions in meters
            must have shape (?, 3)

        Returns:
            elevation in radians (will have same length as sat_pos)
        """
        return self.scenario.station_el(self.station, sat_pos)

    def __contains__(self, tick: int) -> bool:
        """
        Convenience function: `tick in connection` is true iff tick is in
        the range [self.tick0, self.tickn]

        Args:
            tick: the tick to check

        Returns:
            boolean whether or not the tick is included
        """
        return self.tick0 <= tick <= self.tickn

    def _correct_ambiguities_avg(self) -> None:
        """
        Code phase smoothing for carrier phase offsets

        This is the simplest method: use the average difference between
        the code and carrier phases.
        """
        chan2 = self.channel2

        f1, f2 = self.frequencies
        # sign reversal here is correct: ionospheric effect is opposite for code phase
        code_phase_diffs = self.observations[chan2] - self.observations["C1C"]
        carrier_phase_diffs = tec.C * (
            self.observations["L1C"] / f1 - self.observations["L2C"] / f2
        )
        difference = code_phase_diffs - carrier_phase_diffs
        assert abs(numpy.mean(difference)) < 100
        self.offset = numpy.mean(difference)
        self.offset_error = numpy.std(difference)

    def correct_ambiguities(self) -> None:
        """
        Attempt to calculate the offsets from L1C to L2C
        In the complex case by using integer ambiguities
        In the simple case by code phase smoothing
        """
        self._correct_ambiguities_avg()

    @property
    def carrier_correction_meters(self) -> float:
        """
        Returns the correction factor for the chan1 chan2 difference
        This may be calculated with integer ambiguity corrections, or
        using code-phase smoothing

        Note: This could be cached, but this calculation is too simple to be worth it
        """
        # if we have integer ambiguity data, use that
        if self.n_chan1 is not None and self.n_chan2 is not None:
            f1, f2 = self.frequencies
            return tec.C * (self.n_chan2 / f2 - self.n_chan1 / f1)

        # otherwise use the code-phase smoothed difference values
        elif self.offset is not None:
            return self.offset

        assert False, "carrier correction attempted with no correction mechanism"

    @property
    def ipps(self) -> types.ECEF_XYZ_LIST:
        """
        The locations where the signals associated with this connection
        penetrate the ionosphere.

        Returns:
            numpy array of XYZ ECEF coordinates in meters of the IPPs
        """
        return tec.ion_locs(
            self.scenario.station_locs[self.station], self.observations["sat_pos"]
        )

    @property
    def vtecs(self) -> numpy.ndarray:
        """
        The vtec values associated with this connection

        Returns:
            numpy array of (
                vtec value in TECu,
                unitless slant_to_vertical factor
            )
        """
        return tec.calculate_vtecs(self)


class ConnTickMap:
    """
    Simple helper class to efficiently convert a tick number back
    into a connection.
    """

    def __init__(self, connections: Iterable[Connection]) -> None:
        self.connections = connections

    def __getitem__(self, tick: int):
        """
        Get the tick for this tick

        Args:
            tick: the tick to fetch a connection for

        Raises KeyError if tick is not in any of the connections
        """
        for con in self.connections:
            if tick in con:
                return con
        raise KeyError
