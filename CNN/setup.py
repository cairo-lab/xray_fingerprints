#!/usr/bin/env python
# Copyright 2019 Google LLC
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
# ==============================================================================

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'google-cloud-storage>=1.14.0',
    'msgpack-numpy>=0.4.8',
    'pandas>=0.23.4',
    'torch>=1.13.0',
    'torchvision>=0.14.0',
    'scipy>=1.7.3',
    'scikit-learn>=1.0.2',
    'numpy>=1.21.6',
    'python-json-logger>=0.1.11',
    'pyarrow>=14.0.1',
    'torchinfo>=1.8.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='AI Platform | Training | PyTorch | Structured | Python Package'
)
