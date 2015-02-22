#!/bin/bash
pip install pandas
pip install scikit-learn==0.15.2
pip install pybrain
pip install seaborn
pip install skdata
pip install bson
pip install pymongo
git clone https://github.com/hyperopt/hyperopt.git
cd hyperopt
pip install -e .
git clone https://github.com/hyperopt/hyperopt-sklearn.git
cd hyperopt
pip install -e .