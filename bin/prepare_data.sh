#!/bin/bash

# define vavriables
PROJECT_ROOT=..

TRAINING_DATA=${PROJECT_ROOT}/data/training
TEST_DATA=${PROJECT_ROOT}/data/test

unzip  ${TRAINING_DATA}/higgs_training.zip -d ${TRAINING_DATA}
unzip  ${TRAINING_DATA}/bidding_training.csv.zip -d ${TRAINING_DATA}

unzip ${TEST_DATA}/higgs_test.zip -d ${TEST_DATA}
unzip ${TEST_DATA}/bidding_test.csv.zip -d ${TEST_DATA}