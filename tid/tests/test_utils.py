"""
Test utilities functions
"""

import datetime
import pytest
import tid.util as tu


@pytest.mark.parametrize(
    "start_date",
    [
        datetime.datetime(2019, 6, 12, 0, 0),
        datetime.datetime(2019, 6, 12, 6, 0),
        datetime.datetime(2021, 1, 12, 0, 0),
    ],
)
@pytest.mark.parametrize("days", range(1, 5))
def test_date_range(start_date, days):
    """
    Test the data range function
    """
    if start_date.hour == 0 and start_date.minute == 0:
        expected_len = days
    else:
        expected_len = days + 1

    days = tu.get_dates_in_range(start_date, days * tu.DAYS)
    assert len(days) == expected_len
