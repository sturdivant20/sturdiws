cd src

# cmake
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

# SturDDS - Simple messaging between packages
git clone -b main git@github.com:sturdivant20/sturdds.git sturdds

# SturdIO - Simple IO essentials
git clone -b main git@github.com:sturdivant20/sturdio.git sturdio

# SturdINS - My GNSS-INS navigation software
git clone -b main git@github.com:sturdivant20/sturdins.git sturdins

# SturDR - My GNSS software define receiver
git clone -b sturdr++ git@github.com:sturdivant20/sturdr.git sturdr

# FFTW3 (single and double precision)
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
cd ~/devel/sturdr++