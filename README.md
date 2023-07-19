# Missile TID


This library is designed to look for traveling ionospheric disturbances (TID),
a phenomenon caused by space launch vehicles, and large ballistic missiles (among other things)
when they travel through the ionosphere.

## Setup

Some platform-specific dependencies must be installed first. 

#### Ubuntu (20.04)

If in doubt, check the Dockerfile for an exact build recipe, as that will be 
based on an Ubuntu image.

`sudo apt install gcc g++ libcurl4-openssl-dev libgeos-dev`

### MacOS (Monterey/Ventura)

Most issues with getting up and running on MacOS are related to the Shapely
library, which Cartopy imports. If you have issues installing either of these, 
please see more about the pre-reqs here: https://scitools.org.uk/cartopy/docs/latest/installing.html.
You should however only need the geos library, which can be installed with brew or macports:

`brew install geos`

If you still run into issues when producing the plots, for example with the error:
```bash
OSError: Could not find lib geos_c or load any of its variants
```
you must have a Python executable that is running on the same arch as the geos binaries
you installed using brew. This may mean you have the wrong Anaconda version (`x86_64` vs `arm64`),
as an example.

## Installing

Once dependencies are installed, you should be able to install the requirements with:

`python -m pip install -r requirements.txt`

For development, use `requirements-dev.txt` instead. After that, install directly from within the repository:

`python -m pip install -e ./`

For the time being one final step is required: manually make a copy of the example config
file and rename it as `configuration.yml`. The following command should do this:

`cp config/configuration.yml.example config/configuration.yml`

## Running the demos

There are currently two demos available to produce animations of the TID about an area:
* `demos/vandenburg.py`: Displays an animation showing the detection of a Falcon 9 launch out of Vandenburg, CA on the 12th of June, 2019. 
* `demos/live.py`: Monitors for potential launches near the Korean peninsula.

### Common errors

* In some cases the version of `pycurl` that is installed from the requirements
file has a problem where `libffi` is pointing to the wrong version. Reinstalling
`pycurl` manually via `pip` (or `conda` if using that) without the version specifier
seems to fix this.
* If you receive the error: `free(): invalid size` when producing the animation, 
then you must compile Shapely from source. The command `pip install --force-reinstall 
shapely --no-binary shapely` should work for this. 
    * It may be necessary to install `Cython` prior to the shapely reinstall, in which
        case it can be installed via `pip` or `conda` as usual.

## Contributing

Please refer to `CONTRIBUTING.rst`.

## Authors

Because of the refactor, this branch wiped out a lot of author information.

This code was primarily written by @tylerni7 and @MGNute

Further contribution from @tinfoil-globe, @Tobychev, and @jmccartin
