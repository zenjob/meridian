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

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import input_data
from meridian.data import test_utils
import numpy as np
import pandas as pd
import xarray as xr
import xarray.testing as xrt


class InputDataTest(parameterized.TestCase):

  def setUp(self):
    """Generates `sample` DataArrays."""
    super().setUp()

    self.n_times = 149
    self.n_lagged_media_times = 152
    self.n_geos = 10
    self.n_media_channels = 6
    self.n_rf_channels = 2
    self.n_controls = 3

    self.not_lagged_media = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_media_channels=self.n_media_channels,
    )
    self.not_lagged_media_invalid_time_values = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_media_channels=self.n_media_channels,
        date_format="week-%U",
    )
    self.not_lagged_media_invalid_timestamp_values = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_media_channels=self.n_media_channels,
        explicit_time_index=pd.date_range(
            start="2022-01-01", periods=self.n_times, freq="W"
        ),
    )
    self.lagged_media = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_lagged_media_times,
        n_media_channels=self.n_media_channels,
    )
    self.media_spend = test_utils.random_media_spend_nd_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_channels=self.n_media_channels,
    )
    self.media_spend_1d = test_utils.random_media_spend_nd_da(
        n_media_channels=self.n_media_channels
    )
    self.not_lagged_controls = test_utils.random_controls_da(
        media=self.not_lagged_media,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_controls=self.n_controls,
    )
    self.not_lagged_controls_invalid_time_values = (
        test_utils.random_controls_da(
            media=self.not_lagged_media,
            n_geos=self.n_geos,
            n_times=self.n_times,
            n_controls=self.n_controls,
            date_format="week-%U",
        )
    )
    self.not_lagged_controls_timestamp_values = test_utils.random_controls_da(
        media=self.not_lagged_media,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_controls=self.n_controls,
        explicit_time_index=pd.date_range(
            start="2022-01-01", periods=self.n_times, freq="W"
        ),
    )
    self.lagged_controls = test_utils.random_controls_da(
        media=self.lagged_media,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_controls=self.n_controls,
    )
    self.not_lagged_kpi = test_utils.random_kpi_da(
        media=self.not_lagged_media,
        controls=self.not_lagged_controls,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_channels=self.n_media_channels,
        n_controls=self.n_controls,
    )
    self.lagged_kpi = test_utils.random_kpi_da(
        media=self.lagged_media,
        controls=self.lagged_controls,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_channels=self.n_media_channels,
        n_controls=self.n_controls,
    )
    self.revenue_per_kpi = test_utils.constant_revenue_per_kpi(
        n_geos=self.n_geos, n_times=self.n_times, value=2.2
    )
    self.population = test_utils.random_population(n_geos=self.n_geos)
    self.not_lagged_reach = test_utils.random_reach_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_rf_channels=self.n_rf_channels,
    )
    self.lagged_reach = test_utils.random_reach_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_lagged_media_times,
        n_rf_channels=self.n_rf_channels,
    )
    self.not_lagged_frequency = test_utils.random_frequency_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_rf_channels=self.n_rf_channels,
    )
    self.lagged_frequency = test_utils.random_frequency_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_lagged_media_times,
        n_rf_channels=self.n_rf_channels,
    )
    self.rf_spend = test_utils.random_rf_spend_nd_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_rf_channels=self.n_rf_channels,
    )

  def test_construct_media_only_invalid_media_time_values(self):
    with self.assertRaisesRegex(
        ValueError, expected_regex="Invalid media_time label: week-04"
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media_invalid_time_values,
          media_spend=self.media_spend,
      )

  def test_construct_media_only_invalid_media_time_values_type(self):
    with self.assertRaisesRegex(
        ValueError, expected_regex="Invalid media_time label:"
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media_invalid_timestamp_values,
          media_spend=self.media_spend,
      )

  def test_validate_unique_geos(self):
    media = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_media_channels=self.n_media_channels,
        explicit_geo_names=[
            "geo1",
            "geo2",
            "geo3",
            "geo4",
            "geo5",
            "geo6",
            "geo7",
            "geo8",
            "geo9",
            "geo9",
        ],
    )

    with self.assertRaisesRegex(
        ValueError,
        expected_regex="`geo` names must be unique within the array `media`.",
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=media,
          media_spend=self.media_spend,
          reach=self.not_lagged_reach,
          frequency=self.not_lagged_frequency,
          rf_spend=self.rf_spend,
      )

  def test_validate_kpi_wrong_type(self):
    with self.assertRaisesRegex(
        ValueError,
        expected_regex=(
            "Invalid kpi_type: `wrong_type`; must be one of `revenue` or"
            " `non_revenue`."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type="wrong_type",
          population=self.population,
          revenue_per_kpi=self.revenue_per_kpi,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_validate_kpi_negative_values(self):
    with self.assertRaisesRegex(
        ValueError,
        expected_regex="KPI values must be non-negative.",
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi * -1,
          kpi_type=constants.REVENUE,
          population=self.population,
          revenue_per_kpi=self.revenue_per_kpi,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_validate_revenue_per_kpi_negative_values(self):
    with self.assertRaisesRegex(
        ValueError,
        expected_regex=(
            "Revenue per KPI values must not be all zero or negative."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.REVENUE,
          population=self.population,
          revenue_per_kpi=self.revenue_per_kpi * 0,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_validate_media_channels_duplicate_names(self):
    media = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_media_channels=self.n_media_channels,
        explicit_media_channel_names=[
            "ch_1",
            "ch_2",
            "ch_3",
            "ch_4",
            "ch_5",
            "ch_5",
        ],
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_exception_message=(
            "Channel names across `media_channel`, `rf_channel`,"
            " `organic_media_channel`, `organic_rf_channel`, and"
            " `non_media_channel` must be unique. Channel `ch_5` is present in"
            " multiple channel types: ['media_channel', 'media_channel']."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.REVENUE,
          population=self.population,
          revenue_per_kpi=self.revenue_per_kpi,
          media=media,
          media_spend=self.media_spend,
      )

  def test_validate_rf_channels_duplicate_names(self):
    reach = test_utils.random_reach_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_rf_channels=self.n_rf_channels,
        explicit_rf_channel_names=[
            "rf_ch_1",
            "rf_ch_1",
        ],
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_exception_message=(
            "Channel names across `media_channel`, `rf_channel`,"
            " `organic_media_channel`, `organic_rf_channel`, and"
            " `non_media_channel` must be unique. Channel `rf_ch_1` is present"
            " in multiple channel types: ['rf_channel', 'rf_channel']."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.REVENUE,
          population=self.population,
          revenue_per_kpi=self.revenue_per_kpi,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
          reach=reach,
          frequency=self.not_lagged_frequency,
          rf_spend=self.rf_spend,
      )

  def test_validate_rf_channels_with_duplicate_media_channels(self):
    media = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_media_channels=self.n_media_channels,
        explicit_media_channel_names=[
            "ch_1",
            "ch_2",
            "ch_3",
            "ch_4",
            "ch_5",
            "ch_5",
        ],
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_exception_message=(
            "Channel names across `media_channel`, `rf_channel`,"
            " `organic_media_channel`, `organic_rf_channel`, and"
            " `non_media_channel` must be unique. Channel `ch_5` is present in"
            " multiple channel types: ['media_channel', 'media_channel']."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=media,
          media_spend=self.media_spend,
          reach=self.not_lagged_reach,
          frequency=self.not_lagged_frequency,
          rf_spend=self.rf_spend,
      )

  def test_validate_duplicate_channel_names_across_media_and_rf(self):
    media = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_media_channels=self.n_media_channels,
        explicit_media_channel_names=[
            "ch_1",
            "ch_2",
            "ch_3",
            "ch_4",
            "ch_5",
            "ch_6",
        ],
    )
    reach = test_utils.random_reach_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_rf_channels=self.n_rf_channels,
        explicit_rf_channel_names=[
            "rf_ch_1",
            "ch_2",
        ],
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_exception_message=(
            "Channel names across `media_channel`, `rf_channel`,"
            " `organic_media_channel`, `organic_rf_channel`, and"
            " `non_media_channel` must be unique. Channel `ch_2` is present in"
            " multiple channel types: ['media_channel', 'rf_channel']."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=media,
          media_spend=self.media_spend,
          reach=reach,
          frequency=self.not_lagged_frequency,
          rf_spend=self.rf_spend,
      )

  def test_scenarios_kpi_type_revenue_revenue_per_kpi_set_to_ones(self):
    input_data_test = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.REVENUE,
        population=self.population,
        revenue_per_kpi=None,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )
    n_geos = len(input_data_test.kpi.coords[constants.GEO])
    n_times = len(input_data_test.kpi.coords[constants.TIME])

    ones = np.ones((n_geos, n_times))
    revenue_per_kpi = xr.DataArray(
        ones,
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: input_data_test.geo,
            constants.TIME: input_data_test.time,
        },
        name=constants.REVENUE_PER_KPI,
    )
    xrt.assert_equal(input_data_test.revenue_per_kpi, revenue_per_kpi)

  def test_scenarios_kpi_type_revenue_incorrect_revenue_per_kpi_warning(self):
    with self.assertWarnsRegex(
        UserWarning,
        expected_regex=(
            "Revenue from the `kpi` data is used when `kpi_type`=`revenue`."
            " `revenue_per_kpi` is ignored."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.REVENUE,
          population=self.population,
          revenue_per_kpi=self.revenue_per_kpi,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_scenarios_kpi_type_non_revenue_no_revenue_per_kpi_warning(self):
    with self.assertWarnsRegex(
        UserWarning,
        expected_regex=(
            "Consider setting custom priors, as kpi_type was specified as"
            " `non_revenue` with no `revenue_per_kpi` being set. Otherwise, the"
            " total media contribution prior will be used with `p_mean=0.4` and"
            " `p_sd=0.2`. Further documentation available at"
            " https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-custom#set-total-paid-media-contribution-prior"
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          population=self.population,
          revenue_per_kpi=None,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_construct_media_only_invalid_controls_time_values(self):
    with self.assertRaisesRegex(
        ValueError, expected_regex="Invalid time label: week-04"
    ):
      input_data.InputData(
          controls=self.not_lagged_controls_invalid_time_values,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_construct_media_only_invalid_controls_time_values_type(self):
    with self.assertRaisesRegex(
        ValueError, expected_regex="Invalid time label:"
    ):
      input_data.InputData(
          controls=self.not_lagged_controls_timestamp_values,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_construct_from_random_dataarrays_media_only(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )

    xr.testing.assert_equal(data.kpi, self.not_lagged_kpi)
    xr.testing.assert_equal(data.revenue_per_kpi, self.revenue_per_kpi)
    xr.testing.assert_equal(data.controls, self.not_lagged_controls)
    xr.testing.assert_equal(data.population, self.population)
    xr.testing.assert_equal(data.media, self.not_lagged_media)
    xr.testing.assert_equal(data.media_spend, self.media_spend)
    self.assertIsNone(data.reach)
    self.assertIsNone(data.frequency)
    self.assertIsNone(data.rf_spend)

  def test_construct_from_random_dataarrays_media_only_no_controls(self):
    data = input_data.InputData(
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )

    xr.testing.assert_equal(data.kpi, self.not_lagged_kpi)
    xr.testing.assert_equal(data.revenue_per_kpi, self.revenue_per_kpi)
    self.assertIsNone(data.controls)
    self.assertIsNone(data.control_variable)
    xr.testing.assert_equal(data.population, self.population)
    xr.testing.assert_equal(data.media, self.not_lagged_media)
    xr.testing.assert_equal(data.media_spend, self.media_spend)
    self.assertIsNone(data.reach)
    self.assertIsNone(data.frequency)
    self.assertIsNone(data.rf_spend)

  def test_construct_from_random_dataarrays_rf_only(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )

    xr.testing.assert_equal(data.kpi, self.not_lagged_kpi)
    xr.testing.assert_equal(data.revenue_per_kpi, self.revenue_per_kpi)
    xr.testing.assert_equal(data.controls, self.not_lagged_controls)
    xr.testing.assert_equal(data.population, self.population)
    xr.testing.assert_equal(data.reach, self.not_lagged_reach)
    xr.testing.assert_equal(data.frequency, self.not_lagged_frequency)
    xr.testing.assert_equal(data.rf_spend, self.rf_spend)
    self.assertIsNone(data.media)
    self.assertIsNone(data.media_spend)

  def test_construct_from_random_dataarrays_media_and_rf(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )

    xr.testing.assert_equal(data.kpi, self.not_lagged_kpi)
    xr.testing.assert_equal(data.revenue_per_kpi, self.revenue_per_kpi)
    xr.testing.assert_equal(data.media, self.not_lagged_media)
    xr.testing.assert_equal(data.media_spend, self.media_spend)
    xr.testing.assert_equal(data.controls, self.not_lagged_controls)
    xr.testing.assert_equal(data.population, self.population)
    xr.testing.assert_equal(data.reach, self.not_lagged_reach)
    xr.testing.assert_equal(data.frequency, self.not_lagged_frequency)
    xr.testing.assert_equal(data.rf_spend, self.rf_spend)

  def test_properties_media_only(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )

    xr.testing.assert_equal(data.geo, self.not_lagged_kpi[constants.GEO])
    xr.testing.assert_equal(data.time, self.not_lagged_kpi[constants.TIME])
    self.assertEqual(
        data.time_coordinates.all_dates_str,
        [t for t in self.not_lagged_kpi[constants.TIME]],
    )
    xr.testing.assert_equal(
        data.media_time, self.not_lagged_media[constants.MEDIA_TIME]
    )
    self.assertEqual(
        data.media_time_coordinates.all_dates_str,
        [t for t in self.not_lagged_media[constants.MEDIA_TIME]],
    )
    xr.testing.assert_equal(
        data.media_channel, self.not_lagged_media[constants.MEDIA_CHANNEL]
    )
    xr.testing.assert_equal(
        data.control_variable,
        self.not_lagged_controls[constants.CONTROL_VARIABLE],
    )
    self.assertIsNone(data.rf_channel)

  def test_properties_rf_only(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )

    xr.testing.assert_equal(data.geo, self.not_lagged_kpi[constants.GEO])
    xr.testing.assert_equal(data.time, self.not_lagged_kpi[constants.TIME])
    xr.testing.assert_equal(
        data.media_time, self.not_lagged_reach[constants.MEDIA_TIME]
    )
    xr.testing.assert_equal(
        data.rf_channel, self.not_lagged_reach[constants.RF_CHANNEL]
    )
    xr.testing.assert_equal(
        data.control_variable,
        self.not_lagged_controls[constants.CONTROL_VARIABLE],
    )
    self.assertIsNone(data.media_channel)

  def test_geo_property_returns_strings(self):
    media_spend = test_utils.random_media_spend_nd_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_channels=self.n_media_channels,
        integer_geos=True,
    )
    controls = test_utils.random_controls_da(
        media=self.not_lagged_media,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_controls=self.n_controls,
        integer_geos=True,
    )
    kpi = test_utils.random_kpi_da(
        media=self.not_lagged_media,
        controls=self.not_lagged_controls,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_channels=self.n_media_channels,
        n_controls=self.n_controls,
        integer_geos=True,
    )
    revenue_per_kpi = test_utils.constant_revenue_per_kpi(
        n_geos=self.n_geos,
        n_times=self.n_times,
        value=2.2,
        integer_geos=True,
    )
    population = test_utils.random_population(
        n_geos=self.n_geos, integer_geos=True
    )
    media = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_times,
        n_media_channels=self.n_media_channels,
        integer_geos=True,
    )
    data = input_data.InputData(
        controls=controls,
        kpi=kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=revenue_per_kpi,
        population=population,
        media=media,
        media_spend=media_spend,
    )
    self.assertTrue(all(isinstance(g, str) for g in data.geo.values))
    self.assertEqual(
        data.geo.values.tolist(),
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    )
    self.assertIn("1", data.population.coords[constants.GEO].values)

  def test_properties_media_and_rf(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )

    xr.testing.assert_equal(data.geo, self.not_lagged_kpi[constants.GEO])
    xr.testing.assert_equal(data.time, self.not_lagged_kpi[constants.TIME])
    xr.testing.assert_equal(
        data.media_time, self.not_lagged_media[constants.MEDIA_TIME]
    )
    xr.testing.assert_equal(
        data.media_channel, self.not_lagged_media[constants.MEDIA_CHANNEL]
    )
    xr.testing.assert_equal(
        data.rf_channel, self.not_lagged_reach[constants.RF_CHANNEL]
    )
    xr.testing.assert_equal(
        data.control_variable,
        self.not_lagged_controls[constants.CONTROL_VARIABLE],
    )

  @parameterized.named_parameters(
      dict(testcase_name="geo_time_channel", n_geos=10, n_times=100),
      dict(testcase_name="channel", n_geos=None, n_times=None),
  )
  def test_spend_properties(self, n_geos: int | None, n_times: int | None):
    media_spend = test_utils.random_media_spend_nd_da(
        n_geos=n_geos, n_times=n_times, n_media_channels=self.n_media_channels
    )
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=media_spend,
    )
    self.assertEqual(n_geos is not None, data.media_spend_has_geo_dimension)
    self.assertEqual(n_times is not None, data.media_spend_has_time_dimension)
    self.assertFalse(data.rf_spend_has_geo_dimension)
    self.assertFalse(data.rf_spend_has_time_dimension)

  @parameterized.named_parameters(
      dict(testcase_name="geo_time_channel", n_geos=10, n_times=100),
      dict(testcase_name="channel", n_geos=None, n_times=None),
  )
  def test_rf_spend_properties(self, n_geos: int | None, n_times: int | None):
    rf_spend = test_utils.random_rf_spend_nd_da(
        n_geos=n_geos, n_times=n_times, n_rf_channels=self.n_rf_channels
    )
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=rf_spend,
    )
    self.assertEqual(n_geos is not None, data.rf_spend_has_geo_dimension)
    self.assertEqual(n_times is not None, data.rf_spend_has_time_dimension)
    self.assertFalse(data.media_spend_has_geo_dimension)
    self.assertFalse(data.media_spend_has_time_dimension)

  def test_construct_from_random_dataarrays_lagged(self):
    data = input_data.InputData(
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.lagged_media,
        media_spend=self.media_spend,
        reach=self.lagged_reach,
        frequency=self.lagged_frequency,
        rf_spend=self.rf_spend,
    )

    xr.testing.assert_equal(data.kpi, self.lagged_kpi)
    xr.testing.assert_equal(data.revenue_per_kpi, self.revenue_per_kpi)
    xr.testing.assert_equal(data.media, self.lagged_media)
    xr.testing.assert_equal(data.media_spend, self.media_spend)
    xr.testing.assert_equal(data.controls, self.lagged_controls)
    xr.testing.assert_equal(data.population, self.population)
    xr.testing.assert_equal(data.reach, self.lagged_reach)
    xr.testing.assert_equal(data.frequency, self.lagged_frequency)
    xr.testing.assert_equal(data.rf_spend, self.rf_spend)

  def test_construct_from_random_dataarrays_media_spend_1d(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend_1d,
    )
    xr.testing.assert_equal(data.kpi, self.not_lagged_kpi)
    xr.testing.assert_equal(data.revenue_per_kpi, self.revenue_per_kpi)
    xr.testing.assert_equal(data.media, self.not_lagged_media)
    xr.testing.assert_equal(data.media_spend, self.media_spend_1d)
    xr.testing.assert_equal(data.controls, self.not_lagged_controls)
    xr.testing.assert_equal(data.population, self.population)

  def test_as_dataset_media_and_rf(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )

    dataset = data.as_dataset()

    xr.testing.assert_equal(dataset[constants.KPI], self.not_lagged_kpi)
    xr.testing.assert_equal(
        dataset[constants.REVENUE_PER_KPI], self.revenue_per_kpi
    )
    xr.testing.assert_equal(dataset[constants.MEDIA], self.not_lagged_media)
    xr.testing.assert_equal(dataset[constants.MEDIA_SPEND], self.media_spend)
    xr.testing.assert_equal(
        dataset[constants.CONTROLS], self.not_lagged_controls
    )
    xr.testing.assert_equal(dataset[constants.POPULATION], self.population)
    xr.testing.assert_equal(dataset[constants.REACH], self.not_lagged_reach)
    xr.testing.assert_equal(
        dataset[constants.FREQUENCY], self.not_lagged_frequency
    )
    xr.testing.assert_equal(dataset[constants.RF_SPEND], self.rf_spend)

  def test_as_dataset_media_only(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )

    dataset = data.as_dataset()

    xr.testing.assert_equal(dataset[constants.KPI], self.not_lagged_kpi)
    xr.testing.assert_equal(
        dataset[constants.REVENUE_PER_KPI], self.revenue_per_kpi
    )
    xr.testing.assert_equal(
        dataset[constants.CONTROLS], self.not_lagged_controls
    )
    xr.testing.assert_equal(dataset[constants.POPULATION], self.population)
    xr.testing.assert_equal(dataset[constants.MEDIA], self.not_lagged_media)
    xr.testing.assert_equal(dataset[constants.MEDIA_SPEND], self.media_spend)
    self.assertNotIn(constants.REACH, dataset)
    self.assertNotIn(constants.FREQUENCY, dataset)
    self.assertNotIn(constants.RF_SPEND, dataset)

  def test_as_dataset_rf_only(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )

    dataset = data.as_dataset()

    xr.testing.assert_equal(dataset[constants.KPI], self.not_lagged_kpi)
    xr.testing.assert_equal(
        dataset[constants.REVENUE_PER_KPI], self.revenue_per_kpi
    )
    xr.testing.assert_equal(
        dataset[constants.CONTROLS], self.not_lagged_controls
    )
    xr.testing.assert_equal(dataset[constants.POPULATION], self.population)
    xr.testing.assert_equal(dataset[constants.REACH], self.not_lagged_reach)
    xr.testing.assert_equal(
        dataset[constants.FREQUENCY], self.not_lagged_frequency
    )
    xr.testing.assert_equal(dataset[constants.RF_SPEND], self.rf_spend)
    self.assertNotIn(constants.MEDIA, dataset)
    self.assertNotIn(constants.MEDIA_SPEND, dataset)

  def test_wrong_coordinate_name(self):
    media2 = self.not_lagged_media.rename({constants.GEO: "group"})

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "The dimension list of array 'media' doesn't match any of the"
            " following dimension lists: [['geo', 'media_time',"
            " 'media_channel']]."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=media2,
          media_spend=self.media_spend,
      )

  def test_not_matching_dimensions(self):
    media_spend2 = test_utils.random_media_spend_nd_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_channels=self.n_media_channels - 1,
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "'media_channel' dimensions [6, 5] of arrays ['media', 'media_spend']"
        " don't match.",
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media,
          media_spend=media_spend2,
      )

  def test_validate_different_geo_coords(self):
    kpi = self.not_lagged_kpi.copy()
    population = self.population.copy()
    population.coords[constants.GEO] = [
        "geo10",
        "geo11",
        "geo12",
        "geo13",
        "geo14",
        "geo15",
        "geo16",
        "geo17",
        "geo18",
        "geo19",
    ]

    with self.assertRaisesRegex(
        ValueError,
        expected_regex="`geo` coordinates of array `population` don't match.",
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=population,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_not_matching_dimensions_media_spend_1d(self):
    media_spend_1d_2 = test_utils.random_media_spend_nd_da(
        n_media_channels=self.n_media_channels - 1
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "'media_channel' dimensions [6, 5] of arrays ['media', 'media_spend']"
        " don't match.",
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media,
          media_spend=media_spend_1d_2,
      )

  def test_media_spend_2d_fails(self):
    media_spend_2d = test_utils.random_media_spend_nd_da(
        n_geos=self.n_geos, n_media_channels=self.n_media_channels
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "The dimension list of array 'media_spend' doesn't match any of the"
            " following dimension lists: [['media_channel'], ['geo', 'time',"
            " 'media_channel']]."
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media,
          media_spend=media_spend_2d,
      )

  def test_media_shorter_than_kpi_fails(self):
    media_short = self.not_lagged_media[
        :, (self.n_times - self.n_times + 1) :, :
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "The 'media_time' dimension of the 'media' array (148) cannot be"
            " smaller than the 'time' dimension of the 'kpi' array (149)"
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=media_short,
          media_spend=self.media_spend,
      )

  def test_time_interval_irregular(self):
    kpi = self.not_lagged_kpi.copy()
    # In the `kpi` data array copy, tweak one of the `time` coordinate value so
    # that it is not regularly spaced with other coordinate values.
    old_time_coords = kpi[constants.TIME].values
    new_time_coords = old_time_coords.copy()
    new_time_coords[-2] = (
        datetime.datetime.strptime(old_time_coords[-2], constants.DATE_FORMAT)
        + datetime.timedelta(days=2)
    ).strftime(constants.DATE_FORMAT)
    kpi = kpi.assign_coords({constants.TIME: new_time_coords})

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Time coordinates must be regularly spaced.",
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
      )

  def test_media_time_interval_irregular(self):
    media = self.not_lagged_media.copy()
    # In the `media` data array copy, tweak one of the `media_time` coordinate
    # value so that it is not regularly spaced with other coordinate values.
    old_media_time_coords = media[constants.MEDIA_TIME].values
    new_media_time_coords = old_media_time_coords.copy()
    new_media_time_coords[2] = (
        datetime.datetime.strptime(
            old_media_time_coords[2], constants.DATE_FORMAT
        )
        + datetime.timedelta(days=2)
    ).strftime(constants.DATE_FORMAT)
    media = media.assign_coords({constants.MEDIA_TIME: new_media_time_coords})

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Media time coordinates must be regularly spaced.",
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=media,
          media_spend=self.media_spend,
      )

  def test_reach_shorter_than_kpi_fails(self):
    reach_short = self.not_lagged_reach[
        :, (self.n_times - self.n_times + 1) :, :
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "The 'media_time' dimension of the 'reach' array (148) cannot be"
            " smaller than the 'time' dimension of the 'kpi' array (149)"
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
          reach=reach_short,
          frequency=self.not_lagged_frequency,
          rf_spend=self.rf_spend,
      )

  def test_frequency_shorter_than_kpi_fails(self):
    frequency_short = self.not_lagged_frequency[
        :, (self.n_times - self.n_times + 1) :, :
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "The 'media_time' dimension of the 'frequency' array (148) cannot"
            " be smaller than the 'time' dimension of the 'kpi' array"
            " (149)"
        ),
    ):
      input_data.InputData(
          controls=self.not_lagged_controls,
          kpi=self.not_lagged_kpi,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=self.revenue_per_kpi,
          population=self.population,
          media=self.not_lagged_media,
          media_spend=self.media_spend,
          reach=self.not_lagged_reach,
          frequency=frequency_short,
      )

  def test_get_n_top_largest_geos(self):
    population = xr.DataArray(
        data=[100, 5, 50, 10, 75, 25, 80, 30, 20, 95],
        coords={constants.GEO: test_utils.sample_geos(10)},
        name=constants.POPULATION,
    )
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )

    top_geos = data.get_n_top_largest_geos(3)
    self.assertEqual(top_geos, ["geo_0", "geo_9", "geo_6"])

    top_geos = data.get_n_top_largest_geos(5)
    self.assertEqual(top_geos, ["geo_0", "geo_9", "geo_6", "geo_4", "geo_2"])

  def test_get_all_channels_media_only(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )
    channels = data.get_all_channels()
    self.assertTrue(
        (channels == self.not_lagged_media[constants.MEDIA_CHANNEL]).all()
    )

  def test_get_all_channels_rf_only(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )
    channels = data.get_all_channels()
    self.assertTrue(
        (channels == self.not_lagged_reach[constants.RF_CHANNEL]).all()
    )

  def test_get_all_paid_channels(self):
    data = input_data.InputData(
        media=self.lagged_media,
        media_spend=self.media_spend,
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.lagged_reach,
        frequency=self.lagged_frequency,
        rf_spend=self.rf_spend,
    )
    channels = data.get_all_paid_channels()
    expected_paid_channels = [
        m.item() for m in self.lagged_media[constants.MEDIA_CHANNEL]
    ] + [rf.item() for rf in self.lagged_reach[constants.RF_CHANNEL]]
    self.assertSequenceEqual(
        channels.tolist(),
        expected_paid_channels,
    )

  def test_get_paid_channels_argument_builder(self):
    data = input_data.InputData(
        media=self.lagged_media,
        media_spend=self.media_spend,
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.lagged_reach,
        frequency=self.lagged_frequency,
        rf_spend=self.rf_spend,
    )

    paid_channels_arg_builder = data.get_paid_channels_argument_builder()
    expected_paid_channels = [
        m.item() for m in self.lagged_media[constants.MEDIA_CHANNEL]
    ] + [rf.item() for rf in self.lagged_reach[constants.RF_CHANNEL]]

    self.assertSequenceEqual(
        paid_channels_arg_builder._ordered_coords,
        expected_paid_channels,
    )

  def test_get_paid_media_channels_argument_builder(self):
    data = input_data.InputData(
        media=self.lagged_media,
        media_spend=self.media_spend,
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.lagged_reach,
        frequency=self.lagged_frequency,
        rf_spend=self.rf_spend,
    )

    paid_media_channels_arg_builder = (
        data.get_paid_media_channels_argument_builder()
    )

    expected_paid_media_channels = [
        m.item() for m in self.lagged_media[constants.MEDIA_CHANNEL]
    ]
    self.assertSequenceEqual(
        paid_media_channels_arg_builder._ordered_coords,
        expected_paid_media_channels,
    )

  def test_get_paid_media_channels_argument_builder_no_media_channels(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "There are no media channels in the input data.",
    ):
      data.get_paid_media_channels_argument_builder()

  def test_get_paid_rf_channels_argument_builder(self):
    data = input_data.InputData(
        media=self.lagged_media,
        media_spend=self.media_spend,
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.lagged_reach,
        frequency=self.lagged_frequency,
        rf_spend=self.rf_spend,
    )

    paid_rf_channels_arg_builder = data.get_paid_rf_channels_argument_builder()

    expected_paid_rf_channels = [
        rf.item() for rf in self.lagged_reach[constants.RF_CHANNEL]
    ]
    self.assertSequenceEqual(
        paid_rf_channels_arg_builder._ordered_coords,
        expected_paid_rf_channels,
    )

  def test_get_paid_rf_channels_argument_builder_no_rf_channels(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "There are no RF channels in the input data.",
    ):
      data.get_paid_rf_channels_argument_builder()

  def test_get_all_channels_media_and_rf(self):
    data = input_data.InputData(
        media=self.lagged_media,
        media_spend=self.media_spend,
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.lagged_reach,
        frequency=self.lagged_frequency,
        rf_spend=self.rf_spend,
    )
    channels = data.get_all_channels()
    self.assertLen(
        channels,
        self.n_media_channels + self.n_rf_channels,
    )
    self.assertTrue(
        channel in channels
        for channel in self.lagged_media[constants.MEDIA_CHANNEL]
    )
    self.assertTrue(
        channel in channels
        for channel in self.lagged_reach[constants.RF_CHANNEL]
    )

  def test_get_all_media_and_rf(self):
    data = input_data.InputData(
        media=self.lagged_media,
        media_spend=self.media_spend,
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.lagged_reach,
        frequency=self.lagged_frequency,
        rf_spend=self.rf_spend,
    )
    all_media_and_rf = data.get_all_media_and_rf()
    self.assertEqual(
        all_media_and_rf.shape,
        (
            self.n_geos,
            self.n_lagged_media_times,
            self.n_media_channels + self.n_rf_channels,
        ),
    )

    media_from_rf = self.lagged_reach * self.lagged_frequency
    self.assertTrue(media in all_media_and_rf for media in media_from_rf)
    self.assertTrue(media in all_media_and_rf for media in self.lagged_media)

  def test_get_all_spend(self):
    data = input_data.InputData(
        media=self.lagged_media,
        media_spend=self.media_spend,
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=self.lagged_reach,
        frequency=self.lagged_frequency,
        rf_spend=self.rf_spend,
    )
    total_spend = data.get_total_spend()
    self.assertEqual(
        total_spend.shape,
        (
            self.n_geos,
            self.n_times,
            self.n_media_channels + self.n_rf_channels,
        ),
    )
    self.assertTrue(spend in total_spend for spend in self.rf_spend)
    self.assertTrue(spend in total_spend for spend in self.media_spend)

  def test_allocate_media_spend_basic(self):
    """Tests basic allocation, dimensions, and total spend conservation."""
    data = input_data.InputData(
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        media=self.lagged_media,
        media_spend=xr.DataArray(
            [1000, 2000, 3000, 4000, 5000, 6000],
            coords=[self.lagged_media[constants.MEDIA_CHANNEL].media_channel],
            dims=[constants.MEDIA_CHANNEL],
            name=constants.MEDIA_SPEND,
        ),
    )
    allocated_spend = data.allocated_media_spend

    # 1. Verify dimensions
    self.assertEqual(
        allocated_spend.dims,  # pytype: disable=attribute-error
        (constants.GEO, constants.TIME, constants.MEDIA_CHANNEL),
    )
    self.assertLen(allocated_spend[constants.GEO], self.n_geos)
    self.assertLen(allocated_spend[constants.TIME], self.n_times)
    self.assertLen(
        allocated_spend[constants.MEDIA_CHANNEL], self.n_media_channels
    )
    # Verify time coordinates match kpi time, not media_time
    np.testing.assert_array_equal(
        allocated_spend[constants.TIME].values, self.lagged_kpi.time
    )

    # 2. Verify total spend conservation per channel
    total_allocated = allocated_spend.sum(dim=[constants.GEO, constants.TIME])  # pytype: disable=attribute-error
    xr.testing.assert_allclose(total_allocated, data.media_spend)

  def test_allocate_rf_spend_all_zero_media(self):
    """Tests allocation when all media units across all channels are zero."""
    reach_zeros = xr.DataArray(
        np.zeros((self.n_geos, self.n_lagged_media_times, self.n_rf_channels)),
        coords=[
            self.lagged_reach[constants.GEO].geo,
            self.lagged_reach[constants.MEDIA_TIME].media_time,
            self.lagged_reach[constants.RF_CHANNEL].rf_channel,
        ],
        dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
        name=constants.REACH,
    )
    data = input_data.InputData(
        media=self.lagged_media,
        media_spend=self.media_spend,
        controls=self.lagged_controls,
        kpi=self.lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        population=self.population,
        reach=reach_zeros,
        frequency=self.lagged_frequency,
        rf_spend=xr.DataArray(
            [1000, 2000],
            coords=[self.lagged_reach[constants.RF_CHANNEL].rf_channel],
            dims=[constants.RF_CHANNEL],
            name=constants.RF_SPEND,
        ),
    )

    allocated_spend = data.allocated_rf_spend

    # Verify dimensions are still correct
    self.assertEqual(
        allocated_spend.dims,  # pytype: disable=attribute-error
        (constants.GEO, constants.TIME, constants.RF_CHANNEL),
    )

    # All channels had zero total units, expect all NaN allocation
    self.assertTrue(np.isnan(allocated_spend).all())

  def test_get_aggregated_media_spend_no_cal(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        population=self.population,
        media=self.not_lagged_media,  # media_spend's time dim matches kpi time
        media_spend=self.media_spend,  # self.media_spend has n_times
    )
    result = data.aggregate_media_spend(calibration_period=None)
    expected_sum = self.media_spend.values.sum(axis=(0, 1))
    np.testing.assert_array_almost_equal(result, expected_sum)

  def test_get_aggregated_media_spend_with_cal_mixed(self):
    n_spend_times = self.media_spend.shape[1]
    n_channels = self.media_spend.shape[2]
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        population=self.population,
        media=self.not_lagged_media,
        media_spend=self.media_spend,
    )
    # Create a mixed calibration period
    calibration_period = np.random.choice(
        [True, False], size=(n_spend_times, n_channels)
    )
    result = data.aggregate_media_spend(calibration_period=calibration_period)

    factors = np.where(calibration_period.astype(bool), 1.0, 0.0)
    expected_sum = np.einsum("gtm,tm->m", self.media_spend.values, factors)
    np.testing.assert_array_almost_equal(result, expected_sum)

  def test_get_aggregated_rf_spend_no_cal(self):
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        population=self.population,
        reach=self.not_lagged_reach,  # rf_spend's time dim matches kpi time
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,  # self.rf_spend has n_times
    )
    result = data.aggregate_rf_spend(calibration_period=None)
    expected_sum = self.rf_spend.values.sum(axis=(0, 1))
    np.testing.assert_array_almost_equal(result, expected_sum)

  def test_get_aggregated_rf_spend_with_cal_mixed(self):
    n_spend_times = self.rf_spend.shape[1]
    n_channels = self.rf_spend.shape[2]
    data = input_data.InputData(
        controls=self.not_lagged_controls,
        kpi=self.not_lagged_kpi,
        kpi_type=constants.NON_REVENUE,
        population=self.population,
        reach=self.not_lagged_reach,
        frequency=self.not_lagged_frequency,
        rf_spend=self.rf_spend,
    )
    calibration_period = np.random.choice(
        [True, False], size=(n_spend_times, n_channels)
    )
    result = data.aggregate_rf_spend(calibration_period=calibration_period)
    factors = np.where(calibration_period.astype(bool), 1.0, 0.0)
    expected_sum = np.einsum("gtm,tm->m", self.rf_spend.values, factors)
    np.testing.assert_array_almost_equal(result, expected_sum)


class NonpaidInputDataTest(parameterized.TestCase):
  """Tests for non-paid InputData."""

  def setUp(self):
    super().setUp()

    self.n_times = 149
    self.n_lagged_media_times = 152
    self.n_geos = 10
    self.n_media_channels = 6
    self.n_controls = 3

    self.n_non_media_channels = 2
    self.n_organic_media_channels = 4
    self.n_organic_rf_channels = 1

    self.lagged_media = test_utils.random_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_lagged_media_times,
        n_media_channels=self.n_media_channels,
    )
    self.media_spend = test_utils.random_media_spend_nd_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_channels=self.n_media_channels,
    )
    self.controls = test_utils.random_controls_da(
        media=self.lagged_media,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_controls=self.n_controls,
    )
    self.kpi = test_utils.random_kpi_da(
        media=self.lagged_media,
        controls=self.controls,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_channels=self.n_media_channels,
        n_controls=self.n_controls,
    )
    self.non_media_treatments = test_utils.random_non_media_treatments_da(
        media=self.lagged_media,
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_non_media_channels=self.n_non_media_channels,
    )
    self.lagged_organic_media = test_utils.random_organic_media_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_lagged_media_times,
        n_organic_media_channels=self.n_organic_media_channels,
    )
    self.lagged_organic_reach = test_utils.random_organic_reach_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_lagged_media_times,
        n_organic_rf_channels=self.n_organic_rf_channels,
    )
    self.lagged_organic_frequency = test_utils.random_organic_frequency_da(
        n_geos=self.n_geos,
        n_times=self.n_times,
        n_media_times=self.n_lagged_media_times,
        n_organic_rf_channels=self.n_organic_rf_channels,
    )
    self.revenue_per_kpi = test_utils.constant_revenue_per_kpi(
        n_geos=self.n_geos, n_times=self.n_times, value=2.2
    )
    self.population = test_utils.random_population(n_geos=self.n_geos)

  def test_construct_non_media_from_random_dataarrays_lagged(self):
    data = input_data.InputData(
        controls=self.controls,
        kpi=self.kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        non_media_treatments=self.non_media_treatments,
        population=self.population,
        media=self.lagged_media,
        media_spend=self.media_spend,
        organic_media=self.lagged_organic_media,
        organic_reach=self.lagged_organic_reach,
        organic_frequency=self.lagged_organic_frequency,
    )

    xr.testing.assert_equal(data.kpi, self.kpi)
    xr.testing.assert_equal(data.revenue_per_kpi, self.revenue_per_kpi)
    xr.testing.assert_equal(
        data.non_media_treatments, self.non_media_treatments
    )
    xr.testing.assert_equal(data.media, self.lagged_media)
    xr.testing.assert_equal(data.media_spend, self.media_spend)
    xr.testing.assert_equal(data.organic_media, self.lagged_organic_media)
    xr.testing.assert_equal(data.organic_reach, self.lagged_organic_reach)
    xr.testing.assert_equal(
        data.organic_frequency, self.lagged_organic_frequency
    )
    xr.testing.assert_equal(data.controls, self.controls)
    xr.testing.assert_equal(data.population, self.population)

  def test_get_all_paid_channels(self):
    # Expect that non-paid channels are not included.
    data = input_data.InputData(
        controls=self.controls,
        kpi=self.kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=self.revenue_per_kpi,
        non_media_treatments=self.non_media_treatments,
        population=self.population,
        media=self.lagged_media,
        media_spend=self.media_spend,
        organic_media=self.lagged_organic_media,
        organic_reach=self.lagged_organic_reach,
        organic_frequency=self.lagged_organic_frequency,
    )
    channels = data.get_all_paid_channels()
    expected_paid_channels = [
        m.item() for m in self.lagged_media[constants.MEDIA_CHANNEL]
    ]
    self.assertSequenceEqual(
        channels.tolist(),
        expected_paid_channels,
    )

  def test_get_total_outcome_no_revenue(self):
    """Tests get_total_outcome when revenue_per_kpi is None."""
    data = input_data.InputData(
        controls=self.controls,
        kpi=self.kpi,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=None,  # Explicitly set to None
        non_media_treatments=self.non_media_treatments,
        population=self.population,
        media=self.lagged_media,
        media_spend=self.media_spend,
        organic_media=self.lagged_organic_media,
        organic_reach=self.lagged_organic_reach,
        organic_frequency=self.lagged_organic_frequency,
    )
    total_outcome = data.get_total_outcome()
    expected_outcome = np.sum(self.kpi.values)
    # Use xarray testing for potential floating point comparisons
    self.assertAlmostEqual(total_outcome, expected_outcome)

  def test_get_total_outcome_with_revenue(self):
    """Tests get_total_outcome when revenue_per_kpi is provided."""
    data = input_data.InputData(
        controls=self.controls,
        kpi=self.kpi,
        kpi_type=constants.NON_REVENUE,  # kpi_type shouldn't affect this calc
        revenue_per_kpi=self.revenue_per_kpi,  # Provided
        non_media_treatments=self.non_media_treatments,
        population=self.population,
        media=self.lagged_media,
        media_spend=self.media_spend,
        organic_media=self.lagged_organic_media,
        organic_reach=self.lagged_organic_reach,
        organic_frequency=self.lagged_organic_frequency,
    )
    total_outcome = data.get_total_outcome()
    expected_outcome = np.sum(self.kpi.values * self.revenue_per_kpi.values)
    # Use xarray testing for potential floating point comparisons
    # Need to convert expected_outcome to a 0-d DataArray for assert_allclose
    expected_outcome_da = xr.DataArray(expected_outcome)
    self.assertAlmostEqual(total_outcome, expected_outcome_da)


if __name__ == "__main__":
  absltest.main()
