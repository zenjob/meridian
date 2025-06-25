# Copyright 2025 The Meridian Authors.
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

import datetime as dt

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import time_coordinates
import numpy as np
import pandas as pd
import xarray as xr


_ALL_DATES = [
    "2024-01-01",
    "2024-01-08",
    "2024-01-15",
    "2024-01-22",
    "2024-01-29",
    "2024-02-05",
    "2024-02-12",
    "2024-02-19",
]


class TimeCoordinatesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.all_dates = xr.DataArray(
        data=np.array(_ALL_DATES),
        dims=[constants.TIME],
        coords={
            constants.TIME: (
                [constants.TIME],
                _ALL_DATES,
            ),
        },
    )
    self.coordinates = time_coordinates.TimeCoordinates.from_dates(
        self.all_dates
    )

  def test_constructor_must_have_more_than_one_date(self):
    with self.assertRaisesRegex(
        ValueError,
        "There must be more than one date index",
    ):
      time_coordinates.TimeCoordinates.from_dates(
          pd.DatetimeIndex([np.datetime64("2024-01-01")])
      )

  def test_property_all_dates(self):
    expected_dates = [
        dt.datetime.strptime(date, constants.DATE_FORMAT).date()
        for date in _ALL_DATES
    ]
    self.assertEqual(self.coordinates.all_dates, expected_dates)

  def test_property_all_dates_str(self):
    self.assertEqual(self.coordinates.all_dates_str, _ALL_DATES)

  @parameterized.named_parameters(
      dict(
          testcase_name="non_ascending_times_date_strings",
          all_dates=xr.DataArray(
              data=np.array(["2024-01-01", "2024-01-08", "2024-01-07"]),
              dims=[constants.TIME],
              coords={
                  constants.TIME: (
                      [constants.TIME],
                      ["2024-01-01", "2024-01-08", "2024-01-07"],
                  ),
              },
          ),
      ),
      dict(
          testcase_name="non_ascending_times_datetime_index",
          all_dates=pd.DatetimeIndex([
              np.datetime64("2024-01-01"),
              np.datetime64("2024-01-08"),
              np.datetime64("2024-01-15"),
              np.datetime64("2024-01-08"),
              np.datetime64("2024-01-01"),
              np.datetime64("2024-01-05"),
          ]),
      ),
  )
  def test_init_raises_on_non_ascending_times(
      self, all_dates: time_coordinates.TimeCoordinates
  ):
    with self.assertRaisesRegex(
        ValueError,
        "Time coordinates must be strictly monotonically increasing.",
    ):
      time_coordinates.TimeCoordinates.from_dates(all_dates)

  def test_property_interval_days_weekly(self):
    self.assertEqual(self.coordinates.interval_days, 7)

  def test_property_interval_days_daily(self):
    coordinates = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
            np.datetime64("2024-01-04"),
            np.datetime64("2024-01-05"),
        ]),
    )
    self.assertEqual(coordinates.interval_days, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="weekly_interval_days",
          dates=[
              dt.datetime(2024, 1, 1) + dt.timedelta(days=7 * i)
              for i in range(14)
          ],
          expected_interval_days=7,
      ),
      dict(
          testcase_name="thirty_day_interval_days",
          dates=[
              dt.datetime(2024, 1, 1) + dt.timedelta(days=30 * i)
              for i in range(24)
          ],
          expected_interval_days=30,
      ),
  )
  def test_property_interval_days_regular(
      self, dates: list[dt.date], expected_interval_days
  ):
    coordinates = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex(dates)
    )
    self.assertEqual(coordinates.interval_days, expected_interval_days)

  @parameterized.named_parameters(
      dict(
          testcase_name="monthly_on_the_first_day_of_each_month_one_year",
          dates=[dt.datetime(2024, m, 1) for m in range(1, 13)],
          expected_interval_days=30,
      ),
      dict(
          testcase_name="monthly_on_the_last_day_of_each_month_two_years",
          dates=[
              dt.datetime(2024 + y, m, 1) - dt.timedelta(days=1)  # pylint: disable=g-complex-comprehension
              for y in range(2)
              for m in range(1, 13)
          ],
          expected_interval_days=30,
      ),
      dict(
          testcase_name="monthly_on_the_fifteenth_each_month_four_years",
          dates=[
              dt.datetime(2024 + y, m, 15)  # pylint: disable=g-complex-comprehension
              for y in range(4)
              for m in range(1, 13)
          ],
          expected_interval_days=30,
      ),
      dict(
          testcase_name="quarterly_on_the_first_day_of_each_quarter_one_year",
          dates=[dt.datetime(2024, q * 3 + 1, 1) for q in range(0, 4)],
          expected_interval_days=91,
      ),
      dict(
          testcase_name="yearly_with_leap_day",
          dates=[dt.datetime(2023 + y, 1, 1) for y in range(5)],
          expected_interval_days=365,
      ),
  )
  def test_property_interval_days_fuzzy_regular(
      self,
      dates: list[dt.date],
      expected_interval_days: int,
  ):
    coordinates = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex(dates)
    )
    self.assertEqual(coordinates.interval_days, expected_interval_days)

  @parameterized.named_parameters(
      dict(
          testcase_name="weekly_misses_one_entire_week",
          dates=[
              dt.datetime(2024, 1, 1)
              + dt.timedelta(days=7 * (i + 1 if i >= 4 else i))
              for i in range(20)
          ],
      ),
      dict(
          testcase_name="weekly_one_week_is_half_week",
          dates=[
              dt.datetime(2024, 1, 1)
              + dt.timedelta(days=7 * i + (4 if i == 10 else 0))
              for i in range(25)
          ],
      ),
      dict(
          testcase_name="monthly_one_month_is_off_by_ten_days",
          dates=[
              # 2024/3/10 on March, 15th for the rest.
              dt.date(2024, m, 15) - dt.timedelta(days=(10 if m == 2 else 0))
              for m in range(1, 13)
          ],
      ),
  )
  def test_property_interval_days_fuzzy_irregular(
      self,
      dates: list[dt.date],
  ):
    coordinates = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex(dates)
    )
    with self.assertRaisesRegex(
        ValueError,
        "Time coordinates are not regularly spaced!",
    ):
      _ = coordinates.interval_days

  def test_property_irregular_interval_days(self):
    dates = ["2024-01-01", "2024-01-03", "2024-01-10"]
    all_dates = xr.DataArray(
        data=np.array(dates),
        dims=[constants.TIME],
        coords={
            constants.TIME: ([constants.TIME], dates),
        },
    )
    coordinates = time_coordinates.TimeCoordinates.from_dates(all_dates)

    with self.assertRaisesRegex(
        ValueError,
        "Time coordinates are not regularly spaced!",
    ):
      _ = coordinates.interval_days

  def test_get_selected_dates_selected_interval_is_none(self):
    times = self.coordinates.get_selected_dates(
        selected_interval=None,
    )
    self.assertSameElements(
        [t.strftime(constants.DATE_FORMAT) for t in times], self.all_dates.data
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="selected_interval_tuple_str",
          selected_interval=("2024-01-01", "2024-02-19"),
      ),
      dict(
          testcase_name="selected_interval_tuple_datetime",
          selected_interval=(
              dt.datetime(year=2024, month=1, day=1),
              dt.datetime(year=2024, month=2, day=19),
          ),
      ),
      dict(
          testcase_name="selected_interval_tuple_np_datetime64",
          selected_interval=(
              np.datetime64("2024-01-01"),
              np.datetime64("2024-02-19"),
          ),
      ),
      dict(
          testcase_name="selected_interval_date_interval",
          selected_interval=(
              dt.date(year=2024, month=1, day=1),
              dt.date(year=2024, month=2, day=19),
          ),
      ),
  )
  def test_get_selected_dates_selected_interval_matches_range_of_all_dates(
      self, selected_interval: time_coordinates.DateInterval
  ):
    times = self.coordinates.get_selected_dates(
        selected_interval=selected_interval
    )
    self.assertSameElements(
        [t.strftime(constants.DATE_FORMAT) for t in times], self.all_dates.data
    )

  def test_get_selected_dates_selected_interval_is_not_subset_of_all_dates(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"end_date \(2024-02-26\) must be in the time coordinates!",
    ):
      self.coordinates.get_selected_dates(
          selected_interval=("2024-01-01", "2024-02-26"),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="first_half_of_all_dates",
          selected_interval=("2024-01-01", "2024-01-15"),
          expected_dates=["2024-01-01", "2024-01-08", "2024-01-15"],
      ),
      dict(
          testcase_name="second_half_of_all_dates",
          selected_interval=("2024-02-05", "2024-02-19"),
          expected_dates=["2024-02-05", "2024-02-12", "2024-02-19"],
      ),
      dict(
          testcase_name="middle_of_all_dates",
          selected_interval=("2024-01-22", "2024-02-05"),
          expected_dates=["2024-01-22", "2024-01-29", "2024-02-05"],
      ),
  )
  def test_get_selected_dates_converts_selected_interval_into_list_of_dates(
      self, selected_interval: tuple[str, str], expected_dates: list[str]
  ):
    dates = self.coordinates.get_selected_dates(
        selected_interval=selected_interval,
    )
    expected_dates = [
        dt.datetime.strptime(date, constants.DATE_FORMAT).date()
        for date in expected_dates
    ]
    self.assertEqual(dates, expected_dates)

  @parameterized.named_parameters(
      dict(
          testcase_name="start_and_end",
          start_date=dt.datetime(2024, 1, 8).date(),
          end_date=dt.datetime(2024, 2, 5),
          expected_time_dims=[
              dt.datetime(2024, 1, 8).date(),
              dt.datetime(2024, 1, 15).date(),
              dt.datetime(2024, 1, 22).date(),
              dt.datetime(2024, 1, 29).date(),
              dt.datetime(2024, 2, 5).date(),
          ],
      ),
      dict(
          testcase_name="start_only",
          start_date=dt.datetime(2024, 1, 8).date(),
          end_date=None,
          expected_time_dims=[
              dt.datetime(2024, 1, 8).date(),
              dt.datetime(2024, 1, 15).date(),
              dt.datetime(2024, 1, 22).date(),
              dt.datetime(2024, 1, 29).date(),
              dt.datetime(2024, 2, 5).date(),
              dt.datetime(2024, 2, 12).date(),
              dt.datetime(2024, 2, 19).date(),
          ],
      ),
      dict(
          testcase_name="end_only",
          start_date=None,
          end_date=dt.datetime(2024, 2, 5).date(),
          expected_time_dims=[
              dt.datetime(2024, 1, 1).date(),
              dt.datetime(2024, 1, 8).date(),
              dt.datetime(2024, 1, 15).date(),
              dt.datetime(2024, 1, 22).date(),
              dt.datetime(2024, 1, 29).date(),
              dt.datetime(2024, 2, 5).date(),
          ],
      ),
      dict(
          testcase_name="none",
          start_date=None,
          end_date=None,
          expected_time_dims=None,
      ),
      dict(
          testcase_name="start_and_end_are_entire_range",
          start_date=dt.datetime(2024, 1, 1),
          end_date=dt.datetime(2024, 2, 19),
          expected_time_dims=None,
      ),
  )
  def test_expand_selected_time_dims(
      self, start_date, end_date, expected_time_dims
  ):
    self.assertEqual(
        self.coordinates.expand_selected_time_dims(start_date, end_date),
        expected_time_dims,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="start_not_in_data",
          start_date=dt.datetime(2023, 12, 15).date(),
          end_date=dt.datetime(2024, 1, 8).date(),
          expected_error_message=(
              r"start_date \(2023-12-15\) must be in the time coordinates!"
          ),
      ),
      dict(
          testcase_name="end_not_in_data",
          start_date=dt.datetime(2024, 1, 1),
          end_date=dt.datetime(2024, 3, 11),
          expected_error_message=(
              r"end_date \(2024-03-11\) must be in the time coordinates!"
          ),
      ),
      dict(
          testcase_name="start_after_end",
          start_date="2024-01-29",
          end_date="2024-01-01",
          expected_error_message=(
              r"start_date \(2024-01-29\) must be less than or equal to"
              r" end_date \(2024-01-01\)!"
          ),
      ),
  )
  def test_expand_selected_time_dims_fails(
      self, start_date, end_date, expected_error_message
  ):
    with self.assertRaisesRegex(ValueError, expected_error_message):
      self.coordinates.expand_selected_time_dims(start_date, end_date)


if __name__ == "__main__":
  absltest.main()
