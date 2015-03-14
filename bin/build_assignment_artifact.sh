#!/bin/bash

# scripts which creates the archive for submitting the assignment

cd /tmp
rm -rf /tmp/sshegheva3
git clone https://github.com/sshegheva/bag-of-algorithms /tmp/sshegheva3

cp /tmp/sshegheva3/analysis/TASK2_README.md /tmp/sshegheva3/README.md
cp "/tmp/sshegheva3/analysis/notebooks/RandomizedOptimization.pdf" "/tmp/sshegheva3/sshegheva3-analysis.pdf"

tar -cvf sshegheva3.tar.gz sshegheva3