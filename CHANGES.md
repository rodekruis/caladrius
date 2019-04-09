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