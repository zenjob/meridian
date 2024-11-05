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

"""This file contains an object to store prior distributions.

The `PriorDistribution` object contains distributions for various parameters
used by the Meridian model object.
"""

from __future__ import annotations
from collections.abc import MutableMapping
import dataclasses
from typing import Any
import warnings
from meridian import constants
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


__all__ = [
    'PriorDistribution',
]


@dataclasses.dataclass(kw_only=True)
class PriorDistribution:
  """Contains prior distributions for each model parameter.

  PriorDistribution is a utility class for Meridian. The required shapes of the
  arguments to `PriorDistribution` depend on the modeling options and data
  shapes passed to Meridian. For example, `ec_m` is a parameter that represents
  the half-saturation for each media channel. The `ec_m` argument must have
  either `batch_shape=[]` or `batch_shape` equal to the number of media
  channels. In the case of the former, each media channel gets the same prior.

  An error is raised upon Meridian construction if any prior distribution
  has a shape that cannot be broadcast to the shape designated by the model
  specification.

  The parameter batch shapes are as follows:

  | Parameter             | Batch shape        |
  |-----------------------|--------------------|
  | `knot_values`         | `n_knots`          |
  | `tau_g_excl_baseline` | `n_geos - 1`       |
  | `beta_m`              | `n_media_channels` |
  | `beta_rf`             | `n_rf_channels`    |
  | `eta_m`               | `n_media_channels` |
  | `eta_rf`              | `n_rf_channels`    |
  | `gamma_c`             | `n_controls`       |
  | `xi_c`                | `n_controls`       |
  | `alpha_m`             | `n_media_channels` |
  | `alpha_rf`            | `n_rf_channels`    |
  | `ec_m`                | `n_media_channels` |
  | `ec_rf`               | `n_rf_channels`    |
  | `slope_m`             | `n_media_channels` |
  | `slope_rf`            | `n_rf_channels`    |
  | `sigma`               | (σ)                |
  | `roi_m`               | `n_media_channels` |
  | `roi_rf`              | `n_rf_channels`    |

  (σ) `n_geos` if `unique_sigma_for_each_geo`, otherwise this is `1`

  Attributes:
    knot_values: Prior distribution on knots for time effects. Default
      distribution is `Normal(0.0, 5.0)`.
    tau_g_excl_baseline: Prior distribution on geo effects, which represent the
      average KPI of each geo relative to the baseline geo. This parameter is
      broadcast to a vector of length `n_geos - 1`, preserving the geo order and
      excluding the `baseline_geo`. After sampling, `Meridian.inference_data`
      includes a modified version of this parameter called `tau_g`, which has
      length `n_geos` and contains a zero in the position corresponding to
      `baseline_geo`. Meridian ignores this distribution if `n_geos = 1`.
      Default distribution is `Normal(0.0, 5.0)`.
    beta_m: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for impression media channels (`beta_gm`). When
      `media_effects_dist` is set to `'normal'`, it is the hierarchical mean.
      When `media_effects_dist` is set to `'log_normal'`, it is the hierarchical
      parameter for the mean of the underlying, log-transformed, `Normal`
      distribution. Meridian ignores this distribution if `use_roi_prior` is
      `True` and uses `roi_m` prior instead. Default distribution is
      `HalfNormal(5.0)`.
    beta_rf: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for reach and frequency media channels
      (`beta_grf`). When `media_effects_dist` is set to `'normal'`, it is the
      hierarchical mean. When `media_effects_dist` is set to `'log_normal'`, it
      is the hierarchical parameter for the mean of the underlying,
      log-transformed, `Normal` distribution. Meridian ignores this distribution
      if `use_roi_prior` is `True` and uses the `roi_m` prior instead. Default
      distribution is `HalfNormal(5.0)`.
    eta_m: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for impression media channels (`beta_gm`). When
      `media_effects_dist` is set to `'normal'`, it is the hierarchical standard
      deviation. When `media_effects_dist` is set to `'log_normal'` it is the
      hierarchical parameter for the standard deviation of the underlying,
      log-transformed, `Normal` distribution. Default distribution is
      `HalfNormal(1.0)`.
    eta_rf: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for RF media channels (`beta_grf`). When
      `media_effects_dist` is set to `'normal'`, it is the hierarchical standard
      deviation. When `media_effects_dist` is set to `'log_normal'` it is the
      hierarchical parameter for the standard deviation of the underlying,
      log-transformed, `Normal` distribution. Default distribution is
      `HalfNormal(1.0)`.
    gamma_c: Prior distribution on the hierarchical mean of `gamma_gc` which is
      the coefficient on control `c` for geo `g`. Hierarchy is defined over
      geos. Default distribution is `Normal(0.0, 5.0)`.
    xi_c: Prior distribution on the hierarchical standard deviation of
      `gamma_gc` which is the coefficient on control `c` for geo `g`. Hierarchy
      is defined over geos. Default distribution is `HalfNormal(5.0)`.
    alpha_m: Prior distribution on the `geometric decay` Adstock parameter for
      media input. Default distribution is `Uniform(0.0, 1.0)`.
    alpha_rf: Prior distribution on the `geometric decay` Adstock parameter for
      RF input. Default distribution is `Uniform(0.0, 1.0)`.
    ec_m: Prior distribution on the `half-saturation` Hill parameter for media
      input. Default distribution is `TruncatedNormal(0.8, 0.8, 0.1, 10)`.
    ec_rf: Prior distribution on the `half-saturation` Hill parameter for RF
      input. Default distribution is `TransformedDistribution(LogNormal(0.7,
      0.4), Shift(0.1))`.
    slope_m: Prior distribution on the `slope` Hill parameter for media input.
      Default distribution is `Deterministic(1.0)`.
    slope_rf: Prior distribution on the `slope` Hill parameter for RF input.
      Default distribution is `LogNormal(0.7, 0.4)`.
    sigma: Prior distribution on the standard deviation of noise. Default
      distribution is `HalfNormal(5.0)`.
    roi_m: Prior distribution on the hierarchical ROI in media input. Meridian
      ignores this distribution if `use_roi_prior` is `False` and uses `beta_m`
      instead. When `use_roi_prior` is `True` then `beta_m` is calculated as a
      deterministic function of `roi_m`, `alpha_m`, `ec_m`, `slope_m`, and the
      spend associated with each media channel (for example, the model is
      reparameterized with `roi_m` in place of `beta_m`). Default distribution
      is `LogNormal(0.2, 0.9)`. When `use_roi_prior` is `True`, `kpi_type` =
      `non_revenue` and `revenue_per_kpi` is not provided the default value for
      `roi_m` will be ignored and the model will assume a total media
      contribution prior.
    roi_rf: Prior distribution on the hierarchical ROI in RF input. Meridian
      ignores this distribution if `use_roi_prior` is `False` and uses `beta_rf`
      instead. When `use_roi_prior` is `True`, then `beta_rf` is calculated as a
      deterministic function of `roi_rf`, `alpha_rf`, `ec_rf`, `slope_rf`, and
      the spend associated with each media channel (for example, the model is
      reparameterized with `roi_rf` in place of `beta_rf`). Default distribution
      is `LogNormal(0.2, 0.9)`. When `use_roi_prior` is `True`, `kpi_type` =
      `non_revenue` and `revenue_per_kpi` is not provided the default value for
      `roi_rf` will be ignored and the model will assume a total media
      contribution prior.
  """

  knot_values: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Normal(
          0.0, 5.0, name=constants.KNOT_VALUES
      ),
  )
  tau_g_excl_baseline: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Normal(
          0.0, 5.0, name=constants.TAU_G_EXCL_BASELINE
      ),
  )
  beta_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.BETA_M
      ),
  )
  beta_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.BETA_RF
      ),
  )
  eta_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          1.0, name=constants.ETA_M
      ),
  )
  eta_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          1.0, name=constants.ETA_RF
      ),
  )
  gamma_c: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Normal(
          0.0, 5.0, name=constants.GAMMA_C
      ),
  )
  xi_c: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.XI_C
      ),
  )
  alpha_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Uniform(
          0.0, 1.0, name=constants.ALPHA_M
      ),
  )
  alpha_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Uniform(
          0.0, 1.0, name=constants.ALPHA_RF
      ),
  )
  ec_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.TruncatedNormal(
          0.8, 0.8, 0.1, 10, name=constants.EC_M
      ),
  )
  ec_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.TransformedDistribution(
          tfp.distributions.LogNormal(0.7, 0.4),
          tfp.bijectors.Shift(0.1),
          name=constants.EC_RF,
      ),
  )
  slope_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Deterministic(
          1.0, name=constants.SLOPE_M
      ),
  )
  slope_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.7, 0.4, name=constants.SLOPE_RF
      ),
  )
  sigma: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.SIGMA
      ),
  )
  roi_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.2, 0.9, name=constants.ROI_M
      ),
  )
  roi_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.2, 0.9, name=constants.ROI_RF
      ),
  )

  def __setstate__(self, state):
    # Override to support pickling.
    def _unpack_distribution_params(
        params: MutableMapping[str, Any],
    ) -> tfp.distributions.Distribution:
      if constants.DISTRIBUTION in params:
        params[constants.DISTRIBUTION] = _unpack_distribution_params(
            params[constants.DISTRIBUTION]
        )
      dist_type = params.pop(constants.DISTRIBUTION_TYPE)
      return dist_type(**params)

    new_state = {}
    for attribute, value in state.items():
      new_state[attribute] = _unpack_distribution_params(value)

    self.__dict__.update(new_state)

  def __getstate__(self):
    # Override to support pickling.
    state = self.__dict__.copy()

    def _pack_distribution_params(
        dist: tfp.distributions.Distribution,
    ) -> MutableMapping[str, Any]:
      params = dist.parameters
      params[constants.DISTRIBUTION_TYPE] = type(dist)
      if constants.DISTRIBUTION in params:
        params[constants.DISTRIBUTION] = _pack_distribution_params(
            dist.distribution
        )
      return params

    for attribute, value in state.items():
      state[attribute] = _pack_distribution_params(value)

    return state

  def has_deterministic_param(
      self, param: tfp.distributions.Distribution
  ) -> bool:
    return hasattr(self, param) and isinstance(
        getattr(self, param).distribution, tfp.distributions.Deterministic
    )

  def broadcast(
      self,
      n_geos: int,
      n_media_channels: int,
      n_rf_channels: int,
      n_controls: int,
      sigma_shape: int,
      n_knots: int,
      is_national: bool,
      set_roi_prior: bool,
      kpi: float,
      total_spend: np.ndarray,
  ) -> PriorDistribution:
    """Returns a new `PriorDistribution` with broadcast distribution attributes.

    Args:
      n_geos: Number of geos.
      n_media_channels: Number of media channels used.
      n_rf_channels: Number of reach and frequency channels used.
      n_controls: Number of controls used.
      sigma_shape: A number describing the shape of the sigma parameter. It's
        either `1` (if `sigma_for_each_geo=False`) or `n_geos` (if
        `sigma_for_each_geo=True`). For more information, see `ModelSpec`.
      n_knots: Number of knots used.
      is_national: A boolean indicator whether the prior distribution will be
        adapted for a national model.
      set_roi_prior: A boolean indicator whether the ROI prior should be set.
      kpi: Sum of the entire KPI across geos and time. Required if
        `set_roi_prior=True`.
      total_spend: Spend per media channel summed across geos and time. Required
        if `set_roi_prior=True`.

    Returns:
      A new `PriorDistribution` broadcast from this prior distribution,
      according to the given data dimensionality.

    Raises:
      ValueError: If custom priors are not set for all channels.
    """

    def _validate_media_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if (
          param.batch_shape.as_list()
          and n_media_channels != param.batch_shape[0]
      ):
        raise ValueError(
            'Custom priors must have length equal to the number of media'
            ' channels, representing a custom prior for each channel. If you'
            " can't determine a custom prior, consider using the default prior"
            ' for that channel.'
        )

    _validate_media_custom_priors(self.roi_m)
    _validate_media_custom_priors(self.alpha_m)
    _validate_media_custom_priors(self.ec_m)
    _validate_media_custom_priors(self.slope_m)
    _validate_media_custom_priors(self.eta_m)
    _validate_media_custom_priors(self.beta_m)

    def _validate_rf_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if (
          param.batch_shape.as_list()
          and n_media_channels != param.batch_shape[0]
      ):
        raise ValueError(
            'Custom priors must have length equal to the number of RF channels,'
            " representing a custom prior for each channel. If you can't"
            ' determine a custom prior, consider using the default prior for'
            ' that channel.'
        )

    _validate_rf_custom_priors(self.roi_rf)
    _validate_rf_custom_priors(self.alpha_rf)
    _validate_rf_custom_priors(self.ec_rf)
    _validate_rf_custom_priors(self.slope_rf)
    _validate_rf_custom_priors(self.eta_rf)
    _validate_rf_custom_priors(self.beta_rf)

    def _validate_control_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if param.batch_shape.as_list() and n_controls != param.batch_shape[0]:
        raise ValueError(
            'Custom priors must have length equal to the number of control'
            ' variables, representing a custom prior for each control variable.'
            " If you can't determine a custom prior, consider using the default"
            ' prior for that variable.'
        )

    _validate_control_custom_priors(self.gamma_c)
    _validate_control_custom_priors(self.xi_c)

    knot_values = tfp.distributions.BatchBroadcast(
        self.knot_values,
        n_knots,
        name=constants.KNOT_VALUES,
    )
    if is_national:
      tau_g_converted = _convert_to_deterministic_0_distribution(
          self.tau_g_excl_baseline
      )
    else:
      tau_g_converted = self.tau_g_excl_baseline
    tau_g_excl_baseline = tfp.distributions.BatchBroadcast(
        tau_g_converted, n_geos - 1, name=constants.TAU_G_EXCL_BASELINE
    )
    beta_m = tfp.distributions.BatchBroadcast(
        self.beta_m, n_media_channels, name=constants.BETA_M
    )
    beta_rf = tfp.distributions.BatchBroadcast(
        self.beta_rf, n_rf_channels, name=constants.BETA_RF
    )
    if is_national:
      eta_m_converted = _convert_to_deterministic_0_distribution(self.eta_m)
      eta_rf_converted = _convert_to_deterministic_0_distribution(self.eta_rf)
    else:
      eta_m_converted = self.eta_m
      eta_rf_converted = self.eta_rf
    eta_m = tfp.distributions.BatchBroadcast(
        eta_m_converted, n_media_channels, name=constants.ETA_M
    )
    eta_rf = tfp.distributions.BatchBroadcast(
        eta_rf_converted, n_rf_channels, name=constants.ETA_RF
    )
    gamma_c = tfp.distributions.BatchBroadcast(
        self.gamma_c, n_controls, name=constants.GAMMA_C
    )
    if is_national:
      xi_c_converted = _convert_to_deterministic_0_distribution(self.xi_c)
    else:
      xi_c_converted = self.xi_c
    xi_c = tfp.distributions.BatchBroadcast(
        xi_c_converted, n_controls, name=constants.XI_C
    )
    alpha_m = tfp.distributions.BatchBroadcast(
        self.alpha_m, n_media_channels, name=constants.ALPHA_M
    )
    alpha_rf = tfp.distributions.BatchBroadcast(
        self.alpha_rf, n_rf_channels, name=constants.ALPHA_RF
    )
    ec_m = tfp.distributions.BatchBroadcast(
        self.ec_m, n_media_channels, name=constants.EC_M
    )
    ec_rf = tfp.distributions.BatchBroadcast(
        self.ec_rf, n_rf_channels, name=constants.EC_RF
    )
    if (
        not isinstance(self.slope_m, tfp.distributions.Deterministic)
        or (np.isscalar(self.slope_m.loc.numpy()) and self.slope_m.loc != 1.0)
        or (
            self.slope_m.batch_shape.as_list()
            and any(x != 1.0 for x in self.slope_m.loc)
        )
    ):
      warnings.warn(
          'Changing the prior for `slope_m` may lead to convex Hill curves.'
          ' This may lead to poor MCMC convergence and budget optimization'
          ' may no longer produce a global optimum.'
      )
    slope_m = tfp.distributions.BatchBroadcast(
        self.slope_m, n_media_channels, name=constants.SLOPE_M
    )
    slope_rf = tfp.distributions.BatchBroadcast(
        self.slope_rf, n_rf_channels, name=constants.SLOPE_RF
    )
    sigma = tfp.distributions.BatchBroadcast(
        self.sigma, sigma_shape, name=constants.SIGMA
    )

    default_distribution = PriorDistribution()
    if set_roi_prior and _distributions_are_equal(
        self.roi_m, default_distribution.roi_m
    ):
      warnings.warn(
          'Consider setting custom ROI priors, as kpi_type was specified as'
          ' `non_revenue` with no `revenue_per_kpi` being set. Otherwise, the'
          ' total media contribution prior will be used with'
          f' `p_mean={constants.P_MEAN}` and `p_sd={constants.P_SD}` . Further'
          ' documentation available at '
          ' https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi#set-total-media-contribution-prior'
      )
      roi_m_converted = _get_total_media_contribution_prior(
          kpi, total_spend, constants.ROI_M
      )
    else:
      roi_m_converted = self.roi_m
    roi_m = tfp.distributions.BatchBroadcast(
        roi_m_converted, n_media_channels, name=constants.ROI_M
    )

    if set_roi_prior and _distributions_are_equal(
        self.roi_rf, default_distribution.roi_rf
    ):
      warnings.warn(
          'Consider setting custom ROI priors, as kpi_type was specified as'
          ' `non_revenue` with no `revenue_per_kpi` being set. Otherwise, the'
          ' total media contribution prior will be used with'
          f' `p_mean={constants.P_MEAN}` and `p_sd={constants.P_SD}` . Further'
          ' documentation available at '
          ' https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi#set-total-media-contribution-prior'
      )
      roi_rf_converted = _get_total_media_contribution_prior(
          kpi, total_spend, constants.ROI_RF
      )
    else:
      roi_rf_converted = self.roi_rf
    roi_rf = tfp.distributions.BatchBroadcast(
        roi_rf_converted, n_rf_channels, name=constants.ROI_RF
    )

    return PriorDistribution(
        knot_values=knot_values,
        tau_g_excl_baseline=tau_g_excl_baseline,
        beta_m=beta_m,
        beta_rf=beta_rf,
        eta_m=eta_m,
        eta_rf=eta_rf,
        gamma_c=gamma_c,
        xi_c=xi_c,
        alpha_m=alpha_m,
        alpha_rf=alpha_rf,
        ec_m=ec_m,
        ec_rf=ec_rf,
        slope_m=slope_m,
        slope_rf=slope_rf,
        sigma=sigma,
        roi_m=roi_m,
        roi_rf=roi_rf,
    )


def _convert_to_deterministic_0_distribution(
    distribution: tfp.distributions.Distribution,
) -> tfp.distributions.Distribution:
  """Converts the given distribution to a `Deterministic(0)` one.

  Args:
    distribution: `tfp.distributions.Distribution` object to be converted to
      `Deterministic(0)` distribution.

  Returns:
    `tfp.distribution.Deterministic(0, distribution.name)`

  Raises:
    Warning: If the argument distribution is not a `Deterministic(0)`
    distribution.
  """
  if (
      not isinstance(distribution, tfp.distributions.Deterministic)
      or distribution.loc != 0
  ):
    warnings.warn(
        'Hierarchical distribution parameters must be deterministically zero'
        f' for national models. {distribution.name} has been automatically set'
        ' to Deterministic(0).'
    )
    return tfp.distributions.Deterministic(loc=0, name=distribution.name)
  else:
    return distribution


def _get_total_media_contribution_prior(
    kpi: float,
    total_spend: np.ndarray,
    name: str,
    p_mean: float = constants.P_MEAN,
    p_sd: float = constants.P_SD,
) -> tfp.distributions.Distribution:
  """Determines ROI priors based on total media contribution.

  Args:
    kpi: Sum of the entire KPI across geos and time.
    total_spend: Spend per media channel summed across geos and time.
    name: Name of the distribution.
    p_mean: Prior mean proportion of KPI incremental due to all media. Default
      value is `0.4`.
    p_sd: Prior standard deviation proportion of KPI incremental to all media.
      Default value is `0.2`.

  Returns:
    A new `Distribution` based on total media contribution.
  """
  roi_mean = p_mean * kpi / np.sum(total_spend)
  roi_sd = p_sd * kpi / np.sqrt(np.sum(np.power(total_spend, 2)))
  lognormal_sigma = tf.cast(
      np.sqrt(np.log(roi_sd**2 / roi_mean**2 + 1)), dtype=tf.float32
  )
  lognormal_mu = tf.cast(
      np.log(roi_mean * np.exp(-(lognormal_sigma**2) / 2)), dtype=tf.float32
  )
  return tfp.distributions.LogNormal(lognormal_mu, lognormal_sigma, name=name)


def _distributions_are_equal(
    a: tfp.distributions.Distribution, b: tfp.distributions.Distribution
) -> bool:
  """Determine if two distributions are equal."""
  if type(a) != type(b):  # pylint: disable=unidiomatic-typecheck
    return False

  a_params = a.parameters.copy()
  b_params = b.parameters.copy()

  if constants.DISTRIBUTION in a_params and constants.DISTRIBUTION in b_params:
    if not _distributions_are_equal(
        a_params[constants.DISTRIBUTION], b_params[constants.DISTRIBUTION]
    ):
      return False
    del a_params[constants.DISTRIBUTION]
    del b_params[constants.DISTRIBUTION]

  if constants.DISTRIBUTION in a_params or constants.DISTRIBUTION in b_params:
    return False

  return a_params == b_params
