# [SturdiWS](https://github.com/sturdivant20/sturdiws/)

C++ workspace for my (Daniel Sturdivant) thesis work.

## Prerequisites
### *Automatic Method*
You can skip this section by simply executing the `dependencies.sh` script, which will automatically install all dependencies and clone the required development repositories into the `src` folder.
```sh
./dependencies.sh
```

### *Manual Method*
First a valid C++20 compiler should be installed. I recommend `clang++-18` which is easy on ubuntu 24.04.
```sh
sudo apt install clang-18 -y
```

I also recommend `clang-format` for complying to the code formatting of this repository.
```sh
sudo apt install clang-format -y
```

This workspaces depends on multiple low-level packages to function smoothly. This includes `cmake`, `spdlog`, `eigen`, `fftw`, and `pybind11`  Make sure these packages are installed on your computer.
```sh
sudo apt install cmake -y
sudo apt install libspdlog-dev -y
sudo apt install libeigen3-dev -y
sudo apt install python3-pybind11 -y
wget http://www.fftw.org/fftw-3.3.10.tar.gz
tar -xf fftw-3.3.10.tar.gz
cd fftw-3.3.10
sudo ./configure --enable-type-prefix --enable-threads
sudo make
sudo make install
sudo make clean
sudo ./configure --enable-float --enable-type-prefix --enable-threads
sudo make
sudo make install
cd ..
rm -rf fftw-3.3.10
rm -rf fftw-3.3.10.tar.gz
```

Next, essential tools from a navigation toolbox `navengine` are cloned into the development folder `src`:
```sh
git clone -b devel git@github.com:navengine/navtools.git src/navtools
git clone -b devel git@github.com:navengine/satutils.git src/satutils
git clone -b devel git@github.com:navengine/navsim.git src/navsim
```

Lastly, the components of my SDR work should also be cloned into the development folder `src`:
```sh
git clone -b main git@github.com:sturdivant20/sturdio.git src/sturdio
git clone -b main git@github.com:sturdivant20/sturdins.git src/sturdins
git clone -b main git@github.com:sturdivant20/sturdr.git src/sturdr
```

## Building
### *C++ Project*
To build the the standalone C++ project use the build script:
```sh
./build.sh
```
You can turn off the python installation by setting the `INSTALL_PYTHON` option to `False` in the `build.sh` file. You can easily adjust the settings for many other installations in this script.

### *Python Project*
To build the python project, first create a virtual environment, and then pip install the modules (replace the "." with the path to the desired module folder):
```sh
python3 -m venv .venv
. .venv/bin/activate
pip install numpy
pip install scipy
pip install pandas
pip install matplotlib
pip install seaborn
pip install folium
pip install ruamel.yaml
pip install PyQt6
pip install PyQt6-WebEngine
pip install PyQt6-Charts
pip install tqdm
pip install setuptools
pip install ./src/navtools/
pip install ./src/satutils/
pip install ./src/navsim/
pip install ./src/sturdio/
pip install ./src/sturdins/
pip install ./src/sturdr/
```
NOTE: The packages must be pip installed in order of dependency!

## Python Implementation
### *Linting*
Linting provides the auto-complete or messages that appear when you hover over a python module, function, class, etc. This should already be applied, but in case it is not, you can simply do it as follows:
```sh
pip install pybind11-stubgen
pybind11-stubgen navtools -o src/navtools/src
pybind11-stubgen satutils -o src/satutils/src
pybind11-stubgen navsim -o src/navsim/src
pybind11-stubgen sturdins -o src/sturdins/src
pybind11-stubgen sturdr -o src/sturdr/src
```

### *Numpy*
`Numpy` is easily used with all of these C++ packages because the memory of the arrays can be directly accessed. However, to do this Numpy must be configured to save arrays in a *columnwise* manner. This is because the `Eigen` backend defines memory allocations in a columnwise manner. This is easy to implement in Numpy by setting the array order to *Fortran*:
```python
import numpy as np
my_array = np.array([[1,2], [3,4]], order="F")
my_zeros = np.zeros((3,3), order="F")
```
