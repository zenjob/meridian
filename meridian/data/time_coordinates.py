# Copyright 2024 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deals with coordinate values in the time dimensions of input data."""

import dataclasses
import datetime
import functools
from typing import TypeAlias

from meridian import constants
import numpy as np
import pandas as pd
import xarray as xr


__all__ = [
    "Date",
    "DateInterval",
    "TimeCoordinates",
    "normalize_date",
    "normalize_date_interval",
]


# A type alias for a polymorphic "date" type.
Date: TypeAlias = str | datetime.datetime | datetime.date | np.datetime64


# A type alias for a polymorphic "date interval" type. In all variants it is
# always a tuple of (start_date, end_date).
DateInterval: TypeAlias = tuple[Date, Date]

# Time coordinates are string labels in an xarray data array in `InputData`, but
# they can also be parsed into numpy/pandas `DatetimeIndex` internally here.
_TimeCoordinateValues: TypeAlias = pd.DatetimeIndex | xr.DataArray


def _to_pandas_datetime_index(times: _TimeCoordinateValues) -> pd.DatetimeIndex:
  """Normalizes the given values into a pandas `DatetimeIndex`."""
  # Coerce into pandas `DatetimeIndex` in all possible cases.
  return pd.to_datetime(times)


def normalize_date(date: Date) -> datetime.date:
  """Normalizes the given date into a `datetime.date`."""
  if isinstance(date, str):
    return datetime.datetime.strptime(date, constants.DATE_FORMAT).date()
  elif isinstance(date, datetime.datetime):
    return date.date()
  elif isinstance(date, datetime.date):
    return date
  elif isinstance(date, np.datetime64):
    return date.astype(datetime.date)
  else:
    raise ValueError(f"Unsupported date value type: {type(date)} for {date}")


def normalize_date_interval(
    date_interval: DateInterval,
) -> tuple[datetime.date, datetime.date]:
  """Normalizes representations of a date interval into a tuple of `date`s.

  A date interval here is a tuple of `[start_date, end_date)` where:

  * `start_date` is inclusive and `end_date` is exclusive.
  * Both are polymorphic, taking the form of either:
    * `datetime.datetime` (only the date component will be used)
    * `datetime.date` (the normalized form)
    * `np.datetime64` (only the date component will be used)
    * `str` (will be parsed as "YYYY-mm-dd" or else throws)

  In all instances, the given date interval will be normalized as a tuple of
  `datetime.date`s.

  Args:
    date_interval: a polymorphic date interval to normalize.

  Returns:
    A tuple of `date`s representing a `[start_date, end_date)` date interval.
  """
  start, end = date_interval
  start = normalize_date(start)
  end = normalize_date(end)
  return (start, end)


def _to_dates_list(times: pd.DatetimeIndex) -> list[datetime.date]:
  return [dt.date() for dt in times]


@dataclasses.dataclass(frozen=True)
class TimeCoordinates:
  """A wrapper around time coordinates in Meridian's input data.

  Meridian models store time coordinates as untyped strings. It treats them as
  labels, and they have no intrinsic meaning to the model other than the
  assumption that they represent some linearly increasing time coordinates.

  This wrapper object performs some additional validation and methods for
  extracting values out of these time coordinates which are treated as numeric
  "date" values.

  Attributes:
    datetime_index: The given time coordinates, parsed as indexable
      `DatetimeIndex`.
    all_dates: The given time coordinates, as a list of Pythonic `datetime.date`
      objects.
    all_dates_str: The given time coordinates, as a list of Meridian-formatted
      date strings. This can be used for the model internals, which treat time
      coordinates as simple labels.
  """

  datetime_index: pd.DatetimeIndex

  @classmethod
  def from_dates(
      cls,
      dates: _TimeCoordinateValues,
  ) -> "TimeCoordinates":
    """Creates a `TimeCoordinates` from a polymorphic series of dates.

    Args:
      dates: a polymorphic series of dates; it can either be a pandas
        `DatetimeIndex` or an xarray `DataArray` with "YYYY-mm-dd" string
        labels.

    Returns:
      A normalized `TimeCoordinates` dataclass.
    """
    return cls(datetime_index=_to_pandas_datetime_index(dates))

  def __post_init__(self):
    if not self.datetime_index.is_monotonic_increasing:
      raise ValueError("`all_dates` must be strictly monotonic increasing.")

  @property
  def all_dates(self) -> list[datetime.date]:
    return _to_dates_list(self.datetime_index)

  @property
  def all_dates_str(self) -> list[str]:
    return [
        time.strftime(constants.DATE_FORMAT) for time in self.datetime_index
    ]

  @functools.cached_property
  def interval_days(self) -> int:
    """Returns the interval between two neighboring dates in `all_dates`.

    Raises:
      ValueError if the date index is not regularly spaced.
    """
    # Calculate the difference between consecutive dates, in days.
    diff = self.datetime_index.to_series().diff().dt.days.dropna()

    # Check for regularity.
    if diff.nunique() != 1:
      raise ValueError("`datetime_index` coordinates are not evenly spaced!")

    # Finally, return the mode interval.
    return diff.mode()[0]

  def get_selected_dates(
      self,
      selected_interval: DateInterval | None = None,
  ) -> list[datetime.date]:
    """Creates a sequence of dates containing all points in selected interval.

    Args:
      selected_interval: Tuple of the start and end times, or a `DateInterval`
        proto. If None, `all_dates` is returned.

    Returns:
      A sequence of dates representing the subset of `all_dates` between the
      given start and end dates, as Python's builtin `datetime.date` objects.

    Raises:
      ValueError: If `all_dates` are not strictly ascending.
      ValueError: If `selected_interval` is not a subset of `all_dates`.
    """
    if selected_interval is None:
      return self.all_dates

    selected_dates = normalize_date_interval(selected_interval)
    if selected_dates[0] >= selected_dates[1]:
      raise ValueError(
          "`selected_interval` must be a valid (start, end) date interval."
      )

    if any(sd not in self.all_dates for sd in selected_dates):
      raise ValueError("`selected_interval` should be a subset of `all_dates`.")

    start, end = selected_dates
    start_index = self.all_dates.index(start)
    end_index = self.all_dates.index(end) + 1

    return _to_dates_list(self.datetime_index[start_index:end_index])
