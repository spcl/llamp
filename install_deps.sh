#!/bin/bash
# Install dependencies for the toolchain

sudo apt update
sudo apt install -y build-essential manpages-dev software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-11 g++-11 gfortran-11
sudo apt install -y autoconf automake libtool
sudo apt install -y gengetopt
sudo apt install -y re2c
sudo apt install -y graphviz-dev
sudo apt install -y libunwind-dev
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11
sudo apt install -y python3.11-distutils
sudo apt install -y cmake
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
curl -sS https://bootstrap.pypa.io/get-pip.py | python3