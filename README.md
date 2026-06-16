# exp_fileLooper
file looper for beamtime. Thanks to xaratustrah@github

- `looper_data.py` return PSD for whole data files
- `looper_cut.py` return PSD based on injection only for puyuan 4-channel new devices
- `looper_cut2.py` updated `looper_cut.py` for multiprocess

## Prerequisites
- `Python 3`
- `multiprocess`, `matplotlib`
- [`psd-analysis`](https://github.com/NanaVan/psd-analysis)

## Usage
1. configure your own setting for assigned data file at the file `looper_cfg_data.toml` (for looper_data.py), `run_looper.py` (for `looper_cut.py`), and`run_looper2.py` (for `looper_cut2.py`).
2. processing your looper at the terminal 
```Python
> python3 looper_data.py --config looper_cfg_data.toml
> python3 run_looper.py  # looper_cut.py
> python3 run_looper2.py  # looper_cut2.py
```
