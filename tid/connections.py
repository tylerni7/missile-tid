"""
Connections are periods of continuous lock (and therefore carrier phase offsets)
between satellites and ground stations.
Things to manage those are stored here
"""
from __future__ import annotations  # defer type annotations due to circular stuff
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import numpy

from tid import tec, util

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

    @cached_property
    def frequencies(self) -> Tuple[float, float]:
        """
        The frequencies that correspond to this connection
        """
        frequencies = self.scenario.get_frequencies(self.observations)
        assert frequencies is not None, "Unknown frequencies INSIDE connection object"
        return frequencies

    @cached_property
    def channel2(self) -> str:
        """
        The channel2 name "C2C" or "C2P" associated with this connection's data
        """
        chan2 = util.channel2(self.observations)
        assert chan2, "Unknown channel2 data INSIDE connection object"
        return chan2

    @property
    def ticks(self) -> Iterable[int]:
        """
        Iterator of ticks from tick0 to tickn (inclusive), for convenience
        """
        return range(self.tick0, self.tickn + 1)

    @property
    def observations(self) -> numpy.array:
        """
        Convenience function: returns the numpy arrays for the raw observations
        corresponding to this connection
        """
        # note: don't use self.ticks, `range` vs `slice` is a lot slower
        assert self.scenario.station_data
        return self.scenario.station_data[self.station][self.prn][
            self.tick0 : self.tickn + 1
        ]

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
        chan2 = util.channel2(self.observations)
        frequencies = self.scenario.get_frequencies(self.observations)
        # can't do anything without frequencies, use NaN to indicate failure later on
        if not frequencies:
            self.offset = numpy.nan
            return

        f1, f2 = frequencies
        code_phase_diffs = self.observations[chan2] - self.observations["C1C"]
        carrier_phase_diffs = tec.C * (
            self.observations["L1C"] / f1 - self.observations["L2C"] / f2
        )
        difference = code_phase_diffs - carrier_phase_diffs
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
