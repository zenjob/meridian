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

from collections.abc import Collection
from absl.testing import absltest
from absl.testing import parameterized
from meridian.model import knots
import numpy as np


class L1DistanceWeightsTest(parameterized.TestCase):
  """Tests for knots.l1_distance_weights()."""

  @parameterized.named_parameters(
      (
          "knots at 0 and 2",
          np.array([0, 2]),
          np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]),
      ),
      (
          "knots at 0 and 1",
          np.array([0, 1]),
          np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
      ),
      (
          "knots at 1 and 2",
          np.array([1, 2]),
          np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
      ),
      ("full knots", np.array([0, 1, 2]), np.eye(3)),
  )
  def test_correctness(self, knot_locations, expected):
    """Check correctness."""
    knot_locations = knot_locations.astype(np.int32)
    result = knots.l1_distance_weights(3, knot_locations)
    self.assertTrue(np.allclose(result, expected))

  @parameterized.named_parameters(
      ("knots not 1D", np.array([[1, 5]]), "must be one-dimensional"),
      ("knots not sorted", np.array([5, 1]), "must be sorted"),
      ("at least two knots", np.array([5]), "must be greater than 1"),
      ("knots not unique", np.array([1, 5, 5]), "must be unique"),
      ("negative knot", np.array([-1, 5]), "must be positive"),
      ("all negative knots", np.array([-5, -1]), "must be positive"),
      ("knot too big", np.array([5, 10]), "must be less than `n_times`"),
      ("all knots too big", np.array([10, 11]), "must be less than `n_times`"),
  )
  def test_error_checking(self, knot_locations, error_str):
    """Tests that proper errors are raised with impermissible arguments."""
    knot_locations = knot_locations.astype(np.int32)
    with self.assertRaisesRegex(ValueError, error_str):
      knots.l1_distance_weights(10, knot_locations)


class GetKnotInfoTest(parameterized.TestCase):
  """Tests for knots.get_knot_info()."""

  @parameterized.named_parameters(
      dict(
          testcase_name="list_no_right_endpoint",
          n_times=5,
          knots_param=[0, 2, 3],
          is_national=False,
          expected_n_knots=3,
          expected_knot_locations=[0, 2, 3],
          expected_weights=np.array([
              [1, 0.5, 0, 0, 0],
              [0, 0.5, 1, 0, 0],
              [0, 0, 0, 1, 1],
          ]),
      ),
      dict(
          testcase_name="list_no_left_endpoint",
          n_times=5,
          knots_param=[2, 4],
          is_national=False,
          expected_n_knots=2,
          expected_knot_locations=[2, 4],
          expected_weights=np.array([
              [1, 1, 1, 0.5, 0],
              [0, 0, 0, 0.5, 1],
          ]),
      ),
      dict(
          testcase_name="list_no_endpoints",
          n_times=5,
          knots_param=[2, 3],
          is_national=False,
          expected_n_knots=2,
          expected_knot_locations=[2, 3],
          expected_weights=np.array([
              [1, 1, 1, 0, 0],
              [0, 0, 0, 1, 1],
          ]),
      ),
      dict(
          testcase_name="list_both_endpoints",
          n_times=5,
          knots_param=[0, 2, 4],
          is_national=False,
          expected_n_knots=3,
          expected_knot_locations=[0, 2, 4],
          expected_weights=np.array([
              [1, 0.5, 0, 0, 0],
              [0, 0.5, 1, 0.5, 0],
              [0, 0, 0, 0.5, 1],
          ]),
      ),
      dict(
          testcase_name="list_with_gap",
          n_times=7,
          knots_param=[0, 1, 6],
          is_national=False,
          expected_n_knots=3,
          expected_knot_locations=[0, 1, 6],
          expected_weights=np.array([
              [1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0.8, 0.6, 0.4, 0.2, 0],
              [0, 0, 0.2, 0.4, 0.6, 0.8, 1],
          ]),
      ),
      dict(
          testcase_name="int",
          n_times=6,
          knots_param=3,
          is_national=False,
          expected_n_knots=3,
          expected_knot_locations=[0, 2, 5],
          expected_weights=np.array([
              [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.5, 1.0, 0.6666667, 0.33333334, 0.0],
              [0.0, 0.0, 0.0, 0.33333334, 0.6666667, 1.0],
          ]),
      ),
      dict(
          testcase_name="int_same_as_n_times",
          n_times=6,
          knots_param=6,
          is_national=False,
          expected_n_knots=6,
          expected_knot_locations=[0, 1, 2, 3, 4, 5],
          expected_weights=np.array([
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          ]),
      ),
      dict(
          testcase_name="none",
          n_times=4,
          knots_param=None,
          is_national=False,
          expected_n_knots=4,
          expected_knot_locations=[0, 1, 2, 3],
          expected_weights=np.array([
              [1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
          ]),
      ),
      dict(
          testcase_name="none_and_national",
          n_times=6,
          knots_param=None,
          is_national=True,
          expected_n_knots=1,
          expected_knot_locations=[0],
          expected_weights=np.array([
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          ]),
      ),
  )
  def test_sample_list_of_knots(
      self,
      n_times,
      knots_param,
      is_national,
      expected_n_knots,
      expected_knot_locations,
      expected_weights,
  ):
    match knots.get_knot_info(
        n_times=n_times, knots=knots_param, is_national=is_national
    ):
      case knots.KnotInfo(n_knots, knot_locations, weights):
        self.assertEqual(n_knots, expected_n_knots)
        self.assertTrue((knot_locations == expected_knot_locations).all())
        self.assertTrue(np.allclose(weights, expected_weights))
      case _:
        self.fail("Unexpected return type.")

  @parameterized.named_parameters(
      dict(
          testcase_name="too_many",
          knots_arg=201,
          is_national=False,
          msg=(
              "The number of knots (201) cannot be greater than the number of"
              " time periods in the kpi (200)."
          ),
      ),
      dict(
          testcase_name="less_than_one",
          knots_arg=0,
          is_national=False,
          msg="If knots is an integer, it must be at least 1.",
      ),
      dict(
          testcase_name="same_as_n_times_national",
          knots_arg=200,
          is_national=True,
          msg=(
              "Number of knots (200) must be less than number of time periods"
              " (200) in a nationally aggregated model."
          ),
      ),
      dict(
          testcase_name="negative",
          knots_arg=[-2, 17],
          is_national=False,
          msg="Knots must be all non-negative.",
      ),
      dict(
          testcase_name="too_large",
          knots_arg=[3, 202],
          is_national=False,
          msg="Knots must all be less than the number of time periods.",
      ),
  )
  def test_wrong_knots_fails(
      self,
      knots_arg: int | Collection[int] | None,
      is_national: bool,
      msg: str,
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        msg,
    ):
      knots.get_knot_info(n_times=200, knots=knots_arg, is_national=is_national)


if __name__ == "__main__":
  absltest.main()
