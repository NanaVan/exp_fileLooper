# exp_fileLooper
file looper for beamtime. Thanks to xaratustrah@github

- `looper_data.py` return PSD for whole data files
- `looper_cut.py` return PSD based on injection only for puyuan 4-channel new devices

## Prerequisites
- `Python 3`
- `multiprocess`, `matplotlib`
- [`psd-analysis`](https://github.com/NanaVan/psd-analysis)

## Usage
1. configure your own setting for assigned data file at the file `looper_cfg_data.toml` (for looper_data.py), at header of the file `looper_cut.py`.
2. processing your looper at the terminal 
```Python
> python3 looper_data.py --config looper_cfg_data.toml
> python3 looper_cut.py
```
