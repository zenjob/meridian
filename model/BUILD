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
    name = "model_pkg",
    srcs = ["__init__.py"],
    visibility = [
        "//ads/lift/mmm/modeling:__subpackages__",
        "//third_party/py/meridian:__subpackages__",
    ],
    deps = [
        ":adstock_hill",
        ":knots",
        ":model",
        ":prior_distribution",
        ":spec",
        ":transformers",
    ],
)

pytype_strict_library(
    name = "model",
    srcs = ["model.py"],
    visibility = [
        "//ads/lift/mmm/modeling:__subpackages__",
        "//third_party/py/meridian:__subpackages__",
    ],
    deps = [
        ":adstock_hill",
        ":knots",
        ":prior_distribution",
        ":spec",
        ":transformers",
        "//third_party/py/arviz",
        "//third_party/py/joblib",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/data:input_data",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
    ],
)

pytype_strict_contrib_test(
    name = "model_test",
    timeout = "long",
    srcs = ["model_test.py"],
    data = [
        ":test_data",
    ],
    deps = [
        ":adstock_hill",
        ":knots",
        ":model",
        ":prior_distribution",
        ":spec",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/data:test_utils",
        "//third_party/py/mock",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "prior_distribution",
    srcs = ["prior_distribution.py"],
    visibility = ["//third_party/py/meridian:__pkg__"],
    deps = [
        "//third_party/py/meridian:constants",
        "//third_party/py/tensorflow_probability",
    ],
)

pytype_strict_contrib_test(
    name = "prior_distribution_test",
    srcs = ["prior_distribution_test.py"],
    deps = [
        ":prior_distribution",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/meridian:constants",
        "//third_party/py/tensorflow_probability",
    ],
)

pytype_strict_library(
    name = "spec",
    srcs = ["spec.py"],
    visibility = [
        "//ads/lift/mmm/modeling:__subpackages__",
        "//third_party/py/meridian:__subpackages__",
    ],
    deps = [
        ":prior_distribution",
        "//third_party/py/meridian:constants",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "spec_test",
    srcs = ["spec_test.py"],
    deps = [
        ":prior_distribution",
        ":spec",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "transformers",
    srcs = ["transformers.py"],
    visibility = ["//third_party/py/meridian:__subpackages__"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_contrib_test(
    name = "transformers_test",
    srcs = ["transformers_test.py"],
    deps = [
        ":transformers",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
    ],
)

pytype_strict_library(
    name = "adstock_hill",
    srcs = ["adstock_hill.py"],
    visibility = ["//third_party/py/meridian:__subpackages__"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_contrib_test(
    name = "adstock_hill_test",
    srcs = ["adstock_hill_test.py"],
    deps = [
        ":adstock_hill",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
    ],
)

pytype_strict_library(
    name = "knots",
    srcs = ["knots.py"],
    visibility = ["//third_party/py/meridian:__pkg__"],
    deps = [
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "knots_test",
    srcs = ["knots_test.py"],
    deps = [
        ":knots",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

exports_files(["test_data"])
