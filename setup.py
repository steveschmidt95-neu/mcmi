# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mcmi',
    version='0.1.0',
    description='Package for Multiple Class Multiple Instance segmentation of tissues imaged with Mass Spectrometry',
    long_description=readme,
    author='Stephen Schmidt',
    author_email='spschmidt@mgh.harvard.edu',
    url='https://github.com/steveschmidt95-neu/mcmi',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)