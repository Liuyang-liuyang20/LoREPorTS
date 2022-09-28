# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:05:05 2022

@author: liuyang20
"""

from setuptools import setup
from os import path

setup(
    name='LoREPorTS',
    version='0.0.2',
    packages=['LoREPorTS'],
    description="Local Resistance Equivalence-based Pore-Throat Segmentation ",
    long_description_content_type='text/markdown',
    license='MIT',
    keywords=['pore-throat','segmentation network','extraction porespy'],
    author= ['Moran Wang','Yang Liu','Wenbo Gong'],
    url='https://github.com/Liuyang-liuyang20/LoREPorTS',
    download_url='https://github.com/Liuyang-liuyang20/LoREPorTS/releases',
    install_requires=['porespy', 'scipy', 'numpy', 'scikit-image', 'edt'],
    python_requires='>=3',
    
)



