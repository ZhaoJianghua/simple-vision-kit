#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Software: PyCharm
@Author:  zhaojianghua
@Email: zhaojianghua1990@qq.com
@Homepage: None
"""

from setuptools import setup, find_packages
import os


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_PROJECT_ROOT, 'requirements.txt')) as f:
    install_requires = [x.strip() for x in f.readlines() if x.strip()]

with open(os.path.join(_PROJECT_ROOT, "README.md")) as f:
    long_description = f.read()

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

packages = find_packages(where="./src")

setup(
    name="svkcore",
    version="0.0.3",
    description="A simple deep computer vision toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="zjh",
    author_email="zhaojianghua1990@qq.com",
    url="https://gitee.com/jiang_sir/simple-vision-kit",
    packages=packages,
    classifiers=classifiers,
    license_files=["LICENSE"],
    python_requires=">=3.6",
    install_requires=install_requires,
    package_dir={"": "./src"},
    exclude_package_data={'': ['README.*']},
    package_data={
        '': ['assets/DFPKingGothicGB-Light-2.ttf', 'requirements.txt'],
    }
)

