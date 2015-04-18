#!/bin/bash
cd analysis/notebooks
#ipython nbconvert --to latex Supervised\ Classification.ipynb
#pdflatex Supervised\ Classification.tex
#ipython nbconvert --to latex RandomizedOptimization.ipynb
#pdflatex RandomizedOptimization.tex
#ipython nbconvert --to latex "Unsupervised Learning and Dimensionality Reduction.ipynb"
#pdflatex "Unsupervised Learning and Dimensionality Reduction.tex"

ipython nbconvert --to latex "Markov Decision Processes.ipynb"
pdflatex "Markov Decision Processes.tex"
