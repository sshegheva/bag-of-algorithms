#!/bin/bash

# define vavriables
PROJECT_ROOT=..

TRAINING_DATA=${PROJECT_ROOT}/data

unzip  ${TRAINING_DATA}/higgs_training.zip -d ${TRAINING_DATA}
unzip  ${TRAINING_DATA}/bidding_training.csv.zip -d ${TRAINING_DATA}