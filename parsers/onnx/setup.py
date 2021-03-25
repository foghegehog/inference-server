#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#
import os
import sys
import onnx_tensorrt
from setuptools import setup, find_packages

def no_publish():
    blacklist = ['register']
    for cmd in blacklist:
        if cmd in sys.argv:
            raise RuntimeError("Command \"{}\" blacklisted".format(cmd))


REQUIRED_PACKAGES = [
    "pycuda",
    "numpy",
    "onnx"
]

def main():
    no_publish()
    setup(
        name="onnx_tensorrt",
        version=onnx_tensorrt.__version__,
        description="ONNX-TensorRT - TensorRT backend for running ONNX models",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        url="https://github.com/onnx/onnx-tensorrt",
        author="NVIDIA",
        author_email="svc_tensorrt@nvidia.com",
        classifiers=[
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3',
        ],
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        zip_safe=True,
    )

if __name__ == '__main__':
    main()