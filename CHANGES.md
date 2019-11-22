0.6.2 (YYYY-MM-DD)
------------------
- [] use [bulma](https://bulma.io/) ui
- [] add authentication
- [] modularize the UI components

0.6.1 (2019-11-22)
------------------
- Integrated formatters for Python ([Black](https://black.readthedocs.io/en/stable/) and [flake8](https://gitlab.com/pycqa/flake8)) and javascript/css/html/json ([Prettier](https://prettier.io/))
- Enforced formatters using [husky](https://github.com/typicode/husky), [lint-staged](https://github.com/okonet/lint-staged) and [pre-commit](https://pre-commit.com/)
- Fixed bugs in interface
- Create/Download Report

0.6.0 (2019-10-14)
------------------
- Added interface backend to access model and dataset
- Interface allows switching models via dropdown
- Removed builds from conda env file
- Removed yarn dependency
- Updated Docker image

0.5.0 (2019-09-22)
------------------
- Added `accuracy_threshold` as input argument
- Fixed batch size 1 bugs
- Removed setup tools installation process
- Increased verbosity of `sint_maarten_2017.py`
- Switched to miniconda
- Updated Docker image

0.4.0 (2019-07-19)
------------------
- Refactored interface to use React components

0.3.1 (2019-08-12)
------------------
- When creating the individual building images using `caladrius_data`,
  now checks for overlap between different drone images and selects the
  best option, discarding any with <90% good pixels

0.3.0 (2019-06-06)
------------------
- Refactored `caladrius_data` entrypoint so that user must specify which
  components of the data preparation should be run
- Added an option to perform a reverse geocode query for building addresses

0.2.1 (2019-04-09)
------------------
- Added administrative region information to the geojson file used for the visualization

0.2.0 (2019-04-09)
------------------
- Made Caladrius an installable Python package
- Restructured project and placed all Python package and interface files
  in the `caladrius` directory
- Created entrypoints `caladrius_data` for creating the dataset
  and `caladrius` for running the model

0.1.1 (2019-03-31)
------------------
- Added a `maxDataPoints` parameter to `run.py`, which limits the size of the
  data sample. To be used primarily for debugging on non-production machines.

0.1.0 (2019-03-22)
------------------
- Initial version
