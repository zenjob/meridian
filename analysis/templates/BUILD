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

load("//third_party/bazel_rules/rules_sass/sass:sass.bzl", "sass_binary")

package(
    default_applicable_licenses = ["//third_party/py/meridian:package_license"],
    default_visibility = ["//third_party/py/meridian/analysis:__subpackages__"],
)

# This build target only builds CSS in internal Google environments (e.g. internal
# Colab). For external use, the .scss file included in `:templates` will be
# compiled by libsass just in time.
sass_binary(
    name = "style_css",
    src = "style.scss",
)

filegroup(
    name = "templates",
    srcs = glob([
        "*.jinja",
        "*.scss",
        "*.html",
    ]),
)
