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

"""MLflow autologging integration for Meridian."""

from typing import Any, Callable

import arviz as az
import meridian
import mlflow
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from meridian.model import model

FLAVOR_NAME = "meridian"


def _log_versions() -> None:
  """Logs Meridian and ArviZ versions."""
  mlflow.log_param("meridian_version", meridian.__version__)
  mlflow.log_param("arviz_version", az.__version__)


@autologging_integration(FLAVOR_NAME)
def autolog(
    disable: bool = False,  # pylint: disable=unused-argument
    silent: bool = False,  # pylint: disable=unused-argument
) -> None:
  """Enables MLflow tracking for Meridian.

  See https://mlflow.org/docs/latest/tracking/

  Args:
    disable: Whether to disable autologging.
    silent: Whether to suppress all event logs and warnings from MLflow.
  """

  def patch_meridian_init(
      original: Callable[..., Any], *args, **kwargs
  ) -> Callable[..., Any]:
    _log_versions()
    return original(*args, **kwargs)

  safe_patch(FLAVOR_NAME, model.Meridian, "__init__", patch_meridian_init)
