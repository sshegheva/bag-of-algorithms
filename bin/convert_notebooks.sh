#!/bin/bash
cd analysis/notebooks
ipython nbconvert --to latex Supervised\ Classification.ipynb
pdflatex Supervised\ Classification.tex
ipython nbconvert --to latex RandomizedOptimization.ipynb
pdflatex RandomizedOptimization.tex

