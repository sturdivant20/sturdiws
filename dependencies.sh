Reset='\033[0m'           # Text Reset
BoldRed='\033[1;31m'         # Red
BoldGreen='\033[1;32m'       # Green
BoldYellow='\033[1;33m'      # Yellow

cd src

# cmake
sudo apt install cmake -y

# clang
sudo apt install clang-18 -y
sudo apt install clang-format -y

# Python 
sudo apt install libpython3-dev

# spdlog
# git clone -b v2.x git@github.com:gabime/spdlog.git spdlog
sudo apt install libspdlog-dev -y

# Eigen3
# git clone -b 3.4 git@gitlab.com:libeigen/eigen.git eigen3
sudo apt install libeigen3-dev -y

# PyBind11
# git clone -b v2.13 git@github.com:pybind/pybind11.git
sudo apt install python3-pybind11 -y

# Navtools - essential navigation toolbox
git clone -b devel git@github.com:navengine/navtools.git navtools

# Satutils - essential satellite navigation toolbox
git clone -b devel git@github.com:navengine/satutils.git satutils

# Navsim - a simple GNSS navigation observables simulator
git clone -b devel git@github.com:navengine/navsim.git navsim

# SturDDS - Simple messaging between packages
git clone -b main git@github.com:sturdivant20/sturdds.git sturdds

# SturdIO - Simple IO essentials
git clone -b main git@github.com:sturdivant20/sturdio.git sturdio

# SturdINS - My GNSS-INS navigation software
git clone -b main git@github.com:sturdivant20/sturdins.git sturdins

# SturDR - My GNSS software define receiver
git clone -b main git@github.com:sturdivant20/sturdr.git sturdr

# FFTW3
if find "/usr/local/lib/" -type f -name "libfftw3*" -print -quit | grep -q . ; then
    echo -e "${BoldGreen}FFTW3 already installed${Reset}"
else 
    echo -e "${BoldRed}FFTW3 installing\033[0m"
    cd ~/Downloads
    wget http://www.fftw.org/fftw-3.3.10.tar.gz
    tar -xf fftw-3.3.10.tar.gz
    cd fftw-3.3.10
    sudo ./configure --enable-type-prefix --enable-threads --enable-shared
    sudo make CFLAGS=-fPIC
    sudo make install
    # sudo make clean
    # sudo ./configure --enable-float --enable-type-prefix --enable-threads
    # sudo make
    # sudo make install
    cd ..
    sudo rm -rf fftw-3.3.10
    sudo rm -rf fftw-3.3.10.tar.gz
fi

# Fast-DDS
if find "/usr/local/lib/" -type f -name "libfastdds*" -print -quit | grep -q . ; then
    echo -e "${BoldGreen}Fast-DDS already installed${Reset}"
else 
    echo -e "${BoldRed}Fast-DDS installing${Reset}"
    sudo apt install libasio-dev libtinyxml2-dev -y
    sudo apt install libssl-dev -y
    sudo apt install libp11-dev -y
    sudo apt install softhsm2 -y
    sudo usermod -a -G softhsm daniel
    sudo apt install libengine-pkcs11-openssl -y
    sudo apt install openjdk-11-jdk -y
    sudo apt install swig4.1

    mkdir ~/devel/Fast-DDS

    cd ~/devel/Fast-DDS
    git clone https://github.com/eProsima/foonathan_memory_vendor.git
    mkdir foonathan_memory_vendor/build
    cd foonathan_memory_vendor/build
    sudo cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ -DBUILD_SHARED_LIBS=ON
    sudo cmake --build . --target install

    cd ~/devel/Fast-DDS
    git clone https://github.com/eProsima/Fast-CDR.git
    mkdir Fast-CDR/build
    cd Fast-CDR/build
    sudo cmake .. 
    sudo cmake --build . --target install

    cd ~/devel/Fast-DDS
    git clone https://github.com/eProsima/Fast-DDS.git
    mkdir Fast-DDS/build
    cd Fast-DDS/build
    sudo cmake .. 
    sudo cmake --build . --target install

    cd ~/devel/Fast-DDS
    git clone https://github.com/eProsima/Fast-DDS-python.git
    mkdir -p Fast-DDS-python/fastdds_python/build
    cd Fast-DDS-python/fastdds_python/build
    sudo cmake .. 
    sudo cmake --build . --target install

    cd ~/devel/Fast-DDS
    mkdir -p src
    cd src
    git clone --recursive https://github.com/eProsima/Fast-DDS-Gen.git fastddsgen
    cd fastddsgen
    ./gradlew assemble

    export LD_LIBRARY_PATH=/usr/local/lib/
    echo 'export LD_LIBRARY_PATH=/usr/local/lib/' >> ~/.bashrc
    export PATH="$PATH:/home/daniel/devel/Fast-DDS/src/fastddsgen/scripts"
    echo 'export PATH="$PATH:/home/daniel/devel/Fast-DDS/src/fastddsgen/scripts"' >> ~/.bashrc
fi

cd ~/devel/sturdiws
