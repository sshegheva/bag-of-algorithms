#!/bin/bash

# define vavriables
PROJECT_ROOT=..

TRAINING_DATA=${PROJECT_ROOT}/data/training
TEST_DATA=${PROJECT_ROOT}/data/test

# setup directories
rm -rf ${TRAINING_DATA}
mkdir -p ${TRAINING_DATA}
rm -rf ${TEST_DATA}
mkdir -p ${TEST_DATA}


# download data
curl https://www.kaggle.com/c/higgs-boson/download/training.zip -o ${TRAINING_DATA}/training.zip
unzip  ${TRAINING_DATA}/training.zip -C ${TRAINING_DATA}
rm -rf ${TRAINING_DATA}/training.zip

wget https://www.kaggle.com/c/higgs-boson/download/test.zip -P ${TEST_DATA}
unzip ${TEST_DATA}/test.zip -C ${TEST_DATA}
rm -rf ${TEST_DATA}/test.zip