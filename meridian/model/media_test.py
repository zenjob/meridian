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

"""Unit tests for media.py."""

from absl.testing import absltest
from meridian.data import test_utils
from meridian.model import media
from meridian.model import spec
import tensorflow as tf


# Data dimensions for sample input.
_N_GEOS = 5
_N_TIMES = 200
_N_MEDIA_TIMES = 203
_N_CONTROLS = 2
_N_MEDIA_CHANNELS = 3
_N_RF_CHANNELS = 2

_INPUT_DATA_WITH_MEDIA_ONLY = (
    test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        seed=0,
    )
)

_INPUT_DATA_WITH_RF_ONLY = (
    test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_rf_channels=_N_RF_CHANNELS,
        seed=0,
    )
)


class MediaTensorsTest(tf.test.TestCase):

  def test_no_media_values(self):
    model_spec = spec.ModelSpec(
        roi_calibration_period=None, rf_roi_calibration_period=None
    )
    media_tensors = media.build_media_tensors(
        _INPUT_DATA_WITH_RF_ONLY, model_spec
    )

    self.assertIsNone(media_tensors.media)
    self.assertIsNone(media_tensors.media_spend)
    self.assertIsNone(media_tensors.media_transformer)
    self.assertIsNone(media_tensors.media_scaled)
    self.assertIsNone(media_tensors.media_counterfactual)
    self.assertIsNone(media_tensors.media_counterfactual_scaled)
    self.assertIsNone(media_tensors.media_spend_counterfactual)

  def test_media_tensors(self):
    media_tensors = media.build_media_tensors(
        _INPUT_DATA_WITH_MEDIA_ONLY,
        spec.ModelSpec(),
    )

    self.assertAllEqual(
        media_tensors.media,
        tf.convert_to_tensor(
            _INPUT_DATA_WITH_MEDIA_ONLY.media, dtype=tf.float32
        ),
    )
    self.assertAllEqual(
        media_tensors.media_spend,
        tf.convert_to_tensor(
            _INPUT_DATA_WITH_MEDIA_ONLY.media_spend, dtype=tf.float32
        ),
    )

  def _get_tensor_shape(self, tensor: tf.Tensor | None) -> tf.TensorShape:
    if tensor is None:
      raise ValueError("Unexpected None tensor")
    return tensor.shape

  def test_scaled_data_shape(self):
    media_tensors = media.build_media_tensors(
        _INPUT_DATA_WITH_MEDIA_ONLY,
        spec.ModelSpec(),
    )

    self.assertAllEqual(
        self._get_tensor_shape(media_tensors.media_scaled),
        self._get_tensor_shape(_INPUT_DATA_WITH_MEDIA_ONLY.media),
        msg=(
            "Shape of `_media_scaled` does not match the shape of `media`"
            " from the input data."
        ),
    )


class RfTensorsTest(tf.test.TestCase):

  def test_no_rf_values(self):
    model_spec = spec.ModelSpec(
        roi_calibration_period=None, rf_roi_calibration_period=None
    )
    rf_tensors = media.build_rf_tensors(_INPUT_DATA_WITH_MEDIA_ONLY, model_spec)

    self.assertIsNone(rf_tensors.reach)
    self.assertIsNone(rf_tensors.frequency)
    self.assertIsNone(rf_tensors.rf_spend)
    self.assertIsNone(rf_tensors.reach_transformer)
    self.assertIsNone(rf_tensors.reach_scaled)
    self.assertIsNone(rf_tensors.reach_counterfactual)
    self.assertIsNone(rf_tensors.reach_counterfactual_scaled)
    self.assertIsNone(rf_tensors.rf_spend_counterfactual)

  def test_rf_tensors(self):
    rf_tensors = media.build_rf_tensors(
        _INPUT_DATA_WITH_RF_ONLY,
        spec.ModelSpec(),
    )

    self.assertAllEqual(
        rf_tensors.reach,
        tf.convert_to_tensor(_INPUT_DATA_WITH_RF_ONLY.reach, dtype=tf.float32),
    )
    self.assertAllEqual(
        rf_tensors.frequency,
        tf.convert_to_tensor(
            _INPUT_DATA_WITH_RF_ONLY.frequency, dtype=tf.float32
        ),
    )
    self.assertAllEqual(
        rf_tensors.rf_spend,
        tf.convert_to_tensor(
            _INPUT_DATA_WITH_RF_ONLY.rf_spend, dtype=tf.float32
        ),
    )

  def _get_tensor_shape(self, tensor: tf.Tensor | None) -> tf.TensorShape:
    if tensor is None:
      raise ValueError("Unexpected None tensor")
    return tensor.shape

  def test_scaled_data_shape(self):
    rf_tensors = media.build_rf_tensors(
        _INPUT_DATA_WITH_RF_ONLY,
        spec.ModelSpec(),
    )

    self.assertAllEqual(
        self._get_tensor_shape(rf_tensors.reach_scaled),
        self._get_tensor_shape(_INPUT_DATA_WITH_RF_ONLY.reach),
        msg=(
            "Shape of `_reach_scaled` does not match the shape of `reach`"
            " from the input data."
        ),
    )


if __name__ == "__main__":
  absltest.main()
