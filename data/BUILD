# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/meridian:package_license"],
)

pytype_strict_library(
    name = "data_pkg",
    srcs = ["__init__.py"],
    visibility = ["//third_party/py/meridian:__pkg__"],
    deps = [
        ":input_data",
        ":load",
    ],
)

pytype_strict_library(
    name = "input_data",
    srcs = ["input_data.py"],
    visibility = ["//third_party/py/meridian:__subpackages__"],
    deps = [
        "//third_party/py/meridian:constants",
        "//third_party/py/numpy",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    visibility = [
        "//third_party/py/meridian:__subpackages__",
    ],
    deps = [
        ":input_data",
        ":load",
        "//third_party/py/immutabledict",
        "//third_party/py/meridian:constants",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "load",
    srcs = ["load.py"],
    visibility = [
        "//ads/lift/mmm/modeling:__subpackages__",
        "//third_party/py/meridian:__subpackages__",
    ],
    deps = [
        ":input_data",
        "//third_party/py/immutabledict",
        "//third_party/py/meridian:constants",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/xarray",
    ],
)

pytype_strict_contrib_test(
    name = "input_data_test",
    srcs = ["input_data_test.py"],
    deps = [
        ":input_data",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/meridian:constants",
        "//third_party/py/xarray",
    ],
)

pytype_strict_contrib_test(
    name = "load_test",
    srcs = ["load_test.py"],
    data = [
        ":sample/expected_national_data.csv",
        ":sample/lagged_sample_data_media_and_rf.csv",
        ":sample/lagged_sample_data_media_only.csv",
        ":sample/lagged_sample_data_rf_only.csv",
        ":sample/sample_data_media_and_rf.csv",
        ":sample/sample_data_media_only.csv",
        ":sample/sample_data_rf_only.csv",
        ":sample/sample_national_data_w_population_w_geo.csv",
        ":sample/sample_national_data_w_population_wo_geo.csv",
        ":sample/sample_national_data_wo_population_w_geo.csv",
        ":sample/sample_national_data_wo_population_wo_geo.csv",
    ],
    deps = [
        ":load",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/meridian:constants",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/xarray",
    ],
)
