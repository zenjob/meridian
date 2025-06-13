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

from unittest import mock

from absl.testing import absltest
from meridian.data import test_utils
from meridian.mlflow import autolog
import mlflow
from meridian.model import model


INPUT_DATA = test_utils.sample_input_data_revenue(n_media_channels=1)


class AutologTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AutologTest, cls).setUpClass()
    cls.mock_log_param = cls.enter_context(
        mock.patch.object(mlflow, "log_param", autospec=True)
    )

  def tearDown(self):
    self.mock_log_param.reset_mock()
    super().tearDown()

  def test_autolog(self):
    autolog.autolog()
    model.Meridian(input_data=INPUT_DATA)
    self.mock_log_param.assert_called()

  def test_autolog_disabled(self):
    autolog.autolog(disable=True)
    model.Meridian(input_data=INPUT_DATA)
    self.mock_log_param.assert_not_called()


if __name__ == "__main__":
  absltest.main()
