from setuptools import setup

with open('VERSION') as version_file:
    version = version_file.read().strip()

setup(
    name='caladrius',
    version=version,
    url="https://github.com/rodekruis/caladrius",
    packages=["caladrius"],
    setup_requires=[],
    tests_require=[],
    install_requires=[],
    entry_points={
        'console_scripts': [
            'caladrius_run_nn=caladrius.run:main',
            'caladrius_setup_data=caladrius.sint_maarten_2017:main',
        ]
    }
)
