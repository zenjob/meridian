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

"""This file contains data transformers for various inputs of Meridian."""
import numpy as np
import tensorflow as tf


class MediaTransformer:
  """Class containing forward & inverse media transformation methods.

  This class stores scale factors computed from per-geo medians of the `media`
  data, normalized by the geo population.
  """

  def __init__(
      self,
      media: tf.Tensor,
      population: tf.Tensor,
  ):
    """Initializer.

    Args:
      media: A tensor of dimension (`n_geos` x `n_media_times` x
        `n_media_channels`) containing the media data, used to compute the scale
        factors.
      population: A tensor of dimension (`n_geos`) containing the population of
        each geo, used to to compute the scale factors.
    """
    population_scaled_media = tf.math.divide_no_nan(
        media, population[:, tf.newaxis, tf.newaxis]
    )
    # Tensor of medians of the positive portion of `media`. Used as a component
    # for scaling.
    self._population_scaled_median_m = np.nanmedian(
        tf.where(population_scaled_media == 0, np.nan, population_scaled_media),
        [0, 1],
    )
    # Tensor of dimensions (`n_geos` x 1) of weights for scaling `metric`.
    self._scale_factors_gm = tf.einsum(
        "g,m->gm", population, self._population_scaled_median_m
    )

  @property
  def population_scaled_median_m(self):
    return self._population_scaled_median_m

  @tf.function(jit_compile=True)
  def forward(self, media: tf.Tensor) -> tf.Tensor:
    """Scales a given `media` tensor using the stored scale factors."""
    return media / self._scale_factors_gm[:, tf.newaxis, :]

  @tf.function(jit_compile=True)
  def inverse(self, media: tf.Tensor) -> tf.Tensor:
    """Scales a given `media` tensor using the inversed stored scale factors."""
    return media * self._scale_factors_gm[:, tf.newaxis, :]


class ControlsTransformer:
  """Class containing forward & inverse controls transformation methods.

  This class stores means and standard deviations of the controls data, used
  to scale a given `controls` tensor.
  """

  def __init__(self, controls: tf.Tensor):
    """Initializer.

    Args:
      controls: A tensor of dimension (`n_geos` x `n_times` x `n_controls`)
        containing the controls data, used to compute the mean and stddev.
    """
    self._means = tf.reduce_mean(controls, axis=(0, 1))
    self._stdevs = tf.math.reduce_std(controls, axis=(0, 1))

  @tf.function(jit_compile=True)
  def forward(self, controls: tf.Tensor) -> tf.Tensor:
    """Scales a given `controls` tensor using the stored coefficients."""
    return tf.math.divide_no_nan(controls - self._means, self._stdevs)

  @tf.function(jit_compile=True)
  def inverse(self, controls: tf.Tensor) -> tf.Tensor:
    """Scales a given `controls` tensor back using the stored coefficients."""
    return controls * self._stdevs + self._means


class KpiTransformer:
  """Class containing forward & inverse KPI transformation methods.

  This class stores coefficients to scale kpi, first by geo and then
  by mean and standard deviation of kpi.
  """

  def __init__(
      self,
      kpi: tf.Tensor,
      population: tf.Tensor,
  ):
    """Initializer.

    Args:
      kpi: A tensor of dimension (`n_geos` x `n_times`) containing the
        kpi data, used to compute the mean and stddev.
      population: A tensor of dimension (`n_geos`) containing the population of
        each geo, used to to compute the population scale factors.
    """
    self._population = population
    population_scaled_kpi = tf.math.divide_no_nan(
        kpi, self._population[:, tf.newaxis]
    )
    self._population_scaled_mean = tf.reduce_mean(population_scaled_kpi)
    self._population_scaled_stdev = tf.math.reduce_std(
        population_scaled_kpi
    )

  @tf.function(jit_compile=True)
  def forward(self, kpi: tf.Tensor) -> tf.Tensor:
    """Scales a given `kpi` tensor using the stored coefficients."""
    return tf.math.divide_no_nan(
        tf.math.divide_no_nan(kpi, self._population[:, tf.newaxis])
        - self._population_scaled_mean,
        self._population_scaled_stdev,
    )

  @tf.function(jit_compile=True)
  def inverse(self, kpi: tf.Tensor) -> tf.Tensor:
    """Scales a given `kpi` tensor back using the stored coefficients."""
    return (
        kpi * self._population_scaled_stdev + self._population_scaled_mean
    ) * self._population[:, tf.newaxis]
