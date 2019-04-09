from setuptools import setup

with open('VERSION') as version_file:
    version = version_file.read().strip()

setup(
    name='caladrius',
    version=version,
    url="https://github.com/rodekruis/caladrius",
    packages=['caladrius', 'caladrius.model'],
    setup_requires=[],
    tests_require=[],
    install_requires=[],
    entry_points={
        'console_scripts': [
            'caladrius=caladrius.run:main',
            'caladrius_data=caladrius.sint_maarten_2017:main',
        ]
    }
)
