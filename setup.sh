#!/bin/bash
WORK_DIR=$(pwd)

if [ ! -d ./software/pythia8312 ]; then
    cd software
    curl -O https://www.pythia.org/download/pythia83/pythia8312.tgz
    tar xfz pythia8312.tgz
    rm pythia8312.tgz
    cd pythia8312
    export CC=gcc
    export CXX=g++
    ./configure --with-python-config=python3-config | tee config.log
    CHECK=$(cat config.log | awk 'END{print $3}')

    if [[ "$CHECK" == *"PYTHON"* ]]; then
        make -j4
    else
        echo "Pythia cannot find the python installation!"
        echo "Please try using a python virtual environment..."
        cd $WORK_DIR
        exit 1
    fi
    cd $WORK_DIR
fi

if [ ! -f ./software/ML_Tutorial_2025/bin/activate ]; then
    cd software
    python3 -m venv ML_Tutorial_2025
    source ./ML_Tutorial_2025/bin/activate
    pip install --upgrade pip
    pip install -r pip_requirements.txt
    ipython3 kernel install --user --name=ML_Tutorial_2025
    cd $WORK_DIR
else
    source ./software/ML_Tutorial_2025/bin/activate
fi
