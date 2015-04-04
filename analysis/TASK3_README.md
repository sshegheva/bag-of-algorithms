Unsupervised Learning And Dimensionality reduction
===================================================

### Environment Setup

 - Project root directory: `sshegheva3`
 
    - `cd sshegheva3`
    
    - feel free to browse through the code which has following structure:
        - README.md (content of this file)
        - sshegheva3-analysis.pdf (analysis paper for the assignment)
        - algo_evaluation (python package containing necessary implementation for analysis)
        - analysis (IPython Notebooks)
        - data (data used in the analysis)
        - bin (directory with helper bash scripts)
 
 - Project depends on certain python libraries (pandas, sklearn, pybrain, seaborn) defined in the initialization script
 
    - `./bin/init.sh`
    
 - Unpack the data required for analysis (higgs and converters)
 
    - `./bin/prepare_data.sh`
    - the script will unpack higgs and converters dataset
    
    
### Running the analysis
    
 - Analysis can be reproduced interactively using IPython Notebook
 
    - export PYTHONPATH variable location where code was unzipped
        (eg. see script bin/start_notebook.sh)
        `export PYTHONPATH=[location-of-unzipped-artifact]/sshegheva3`
 
    - `./bin/start_notebook.sh`
    
    - open http://localhost:8888 to view existing notebooks
    
    - select `Unsupervised Learning and Dimensionality Reduction` notebook and interact with the results
    
    - you can select `Cell->Run All` or execute statements one by one by pressing `Shift+Enter`
    
 - Notebook can subsequently be turned into PDF (you might need additional libraries for this)
 
    - `./bin/convert_notebooks.sh`
    
    - find the `Unsupervised Learning and Dimensionality Reduction.pdf` 