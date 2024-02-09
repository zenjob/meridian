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

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/meridian:package_license"],
)

pytype_strict_library(
    name = "analysis_pkg",
    srcs = ["__init__.py"],
    visibility = ["//third_party/py/meridian:__pkg__"],
    deps = [
        ":analyzer",
        ":formatter",
        ":optimizer",
        ":summarizer_lib",
        ":visualizer",
    ],
)

pytype_strict_library(
    name = "analyzer",
    srcs = ["analyzer.py"],
    deps = [
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/model",
        "//third_party/py/meridian/model:adstock_hill",
        "//third_party/py/meridian/model:transformers",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/xarray",
    ],
)

pytype_strict_contrib_test(
    name = "analyzer_test",
    timeout = "long",
    srcs = ["analyzer_test.py"],
    data = [
        "//third_party/py/meridian/model:test_data",
    ],
    deps = [
        ":analyzer",
        ":test_utils",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/arviz",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/data:test_utils",
        "//third_party/py/meridian/model",
        "//third_party/py/meridian/model:spec",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "visualizer",
    srcs = ["visualizer.py"],
    deps = [
        ":analyzer",
        ":formatter",
        ":summary_text",
        "//third_party/py/altair",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/model",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/xarray",
    ],
)

pytype_strict_contrib_test(
    name = "visualizer_test",
    srcs = ["visualizer_test.py"],
    data = [
        "//third_party/py/meridian/model:test_data",
    ],
    deps = [
        ":analyzer",
        ":formatter",
        ":summary_text",
        ":test_utils",
        ":visualizer",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/altair",
        "//third_party/py/arviz",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/data:input_data",
        "//third_party/py/meridian/model",
        "//third_party/py/numpy",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "summarizer_lib",
    srcs = [
        "summarizer.py",
    ],
    data = [
        "//third_party/py/meridian/analysis/templates",
        "//third_party/py/meridian/analysis/templates:style.css",
    ],
    deps = [
        ":formatter",
        ":summary_text",
        ":visualizer",
        "//base/python:pywrapbase",
        "//third_party/py/jinja2",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/model",
        "//third_party/py/pandas",
        "//third_party/py/xarray",
    ],
)

pytype_strict_contrib_test(
    name = "summarizer_test",
    srcs = ["summarizer_test.py"],
    deps = [
        ":analyzer",
        ":summarizer_lib",
        ":summary_text",
        ":test_utils",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/model",
        "//third_party/py/mock",
        "//third_party/py/xarray",
        "//webutil/html/types/sanitizer/python:html_sanitizer",
    ],
)

pytype_strict_binary(
    name = "summarizer",
    srcs = ["summarizer.py"],
    deps = [":summarizer_lib"],
)

pytype_strict_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    data = [
        "//third_party/py/meridian/analysis/templates",
        "//third_party/py/meridian/analysis/templates:style.css",
    ],
    deps = [
        ":analyzer",
        ":formatter",
        ":summary_text",
        "//third_party/py/altair",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/model",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "formatter",
    srcs = ["formatter.py"],
    data = [
        "//third_party/py/meridian/analysis/templates",
    ],
    deps = [
        "//third_party/py/altair",
        "//third_party/py/immutabledict",
        "//third_party/py/jinja2",
        "//third_party/py/meridian:constants",
    ],
)

pytype_strict_contrib_test(
    name = "formatter_test",
    srcs = ["formatter_test.py"],
    deps = [
        ":formatter",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "summary_text",
    srcs = ["summary_text.py"],
)

pytype_strict_contrib_test(
    name = "optimizer_test",
    srcs = ["optimizer_test.py"],
    data = [
        "//third_party/py/meridian/model:test_data",
    ],
    deps = [
        ":analyzer",
        ":formatter",
        ":optimizer",
        ":summary_text",
        ":test_utils",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/altair",
        "//third_party/py/arviz",
        "//third_party/py/meridian:constants",
        "//third_party/py/meridian/data:input_data",
        "//third_party/py/meridian/data:test_utils",
        "//third_party/py/meridian/model",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/xarray",
    ],
)

pytype_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    visibility = ["//third_party/py/meridian:__subpackages__"],
    deps = [
        "//third_party/py/meridian:constants",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/xarray",
    ],
)
