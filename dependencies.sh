cd src

cmake
sudo apt install cmake -y

# clang
sudo apt install clang -y
sudo apt install clang-format -y

# spdlog
# git clone -b v2.x git@github.com:gabime/spdlog.git spdlog
sudo apt install libspdlog-dev -y

# Eigen3
# git clone -b 3.4 git@gitlab.com:libeigen/eigen.git eigen3
sudo apt install libeigen3-dev -y

# Navtools - essential navigation toolbox
git clone -b main git@github.com:navengine/navtools.git navtools

# Satutils - essential satellite navigation toolbox
git clone -b devel git@github.com:navengine/satutils.git satutils

# # SturDDS - Simple messaging between packages
# git clone -b main git@github.com:sturdivant20/sturdds.git sturdds

# SturdIO - Simple IO essentials
git clone -b main git@github.com:sturdivant20/sturdio.git sturdio

# SturdINS - My GNSS-INS navigation software
git clone -b main git@github.com:sturdivant20/sturdins.git sturdins

# SturDR - My GNSS software define receiver
git clone -b main git@github.com:sturdivant20/sturdr.git sturdr

# FFTW3 (single and double precision)
HAS_FFTW=False
(ls /usr/local/lib/libfftw3* >> /dev/null 2>&1 && HAS_FFTW=True) || HAS_FFTW=False
if [ $HAS_FFTW ]
then 
    echo FFTW3 already installed
else 
    echo FFTW3 installing
    cd ~/Downloads
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
fi

# Fast-DDS
HAS_FAST_DDS=False
(ls /usr/local/lib/libfastdds* >> /dev/null 2>&1 && HAS_FAST_DDS=True) || HAS_FAST_DDS=False
if [ $HAS_FAST_DDS ]
then 
    echo Fast-DDS already installed
else 
    echo Fast-DDS installing
    sudo apt install libasio-dev libtinyxml2-dev
    sudo apt install libssl-dev
    sudo apt install libp11-dev
    sudo apt install softhsm2
    sudo usermod -a -G softhsm daniel
    sudo apt install libengine-pkcs11-openssl
    mkdir ~/devel/Fast-DDS

    cd ~/devel/Fast-DDS
    git clone https://github.com/eProsima/foonathan_memory_vendor.git
    mkdir foonathan_memory_vendor/build
    cd foonathan_memory_vendor/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ -DBUILD_SHARED_LIBS=ON
    cmake --build . --target install

    cd ~/devel/Fast-DDS
    git clone https://github.com/eProsima/Fast-CDR.git
    mkdir Fast-CDR/build
    cd Fast-CDR/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ -DBUILD_SHARED_LIBS=ON
    cmake --build . --target install

    cd ~/devel/Fast-DDS
    git clone https://github.com/eProsima/Fast-DDS.git
    mkdir Fast-DDS/build
    cd Fast-DDS/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ -DBUILD_SHARED_LIBS=ON
    cmake --build . --target install

    sudo apt install openjdk-11-jdk
    cd ~/devel/Fast-DDS
    mkdir -p src
    cd src
    git clone --recursive https://github.com/eProsima/Fast-DDS-Gen.git fastddsgen
    cd fastddsgen
    ./gradlew assemble
fi

cd ~/devel/sturdiws
