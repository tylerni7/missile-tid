# Missile TID

This library aims to do GNSS stuff with the goal of getting ionospheric TEC data.

Right now it is "functional" but only barely.

## Getting started

### Dependencies
You will need a number of libaries to get started. Most of these are python packages, but at lease one of these
has dependencies which you will need to install on your system first.

MacOS:
```shell script
brew install geos
```

Windows/WSL or Debian
```shell script
sudo apt install geos
```

There is also a minimise function, which is C code that needs to be compiled. This can be done in python:
```python
cd pytid
python setup.py install
```
You can then install the remainder of the python libraries via pip:
```shell script
python -m pip install -r requirements.txt
````

Finally, if running from virtual environments, you should add the project root folder to the PYTHONPATH. E.g:
```shell script
export PYTHONPATH=$(pwd)
```

### Configuration
Configuration is done via YAML at the project level. Right now this only affects the station plotter, but will
in time cover the whole repo. Please copy the `configuration.yml.example` found in the config
folder to `configuration.yml`. You will need to set the paths to where the gnss data will be downloaded.
This configuration also lets you set the ground stations for which the vTEC will be plotted, as well as several
other project-level settings that will be added later.

## Generating plots
You can plot the vTEC for each ground station using `station_plotter.py`. This takes a number of optional
arguments, but you must specify a date in the form (YYYY-mm-dd). For example:
```shell script
python pytid/station_plotter.py --start-date 2020-2-17
```
The plots will then be saved to wherever you configured the output to go (`plots` folder by default).
