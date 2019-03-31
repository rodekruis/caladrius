from setuptools import setup

with open('VERSION') as version_file:
    version = version_file.read().strip()

setup(
    name='caladrius',
    version=version,
)
