Supervised Classification
==========================

### Environment Setup

Note: Code and Analysis can also be cloned from the repo: https://github.com/sshegheva/bag-of-algorithms

 - Project root directory: `bag-of-algorithms`
 
    - `cd bag-of-algorithms`
    
    - feel free to browse through the code which has following structure:
        - algo_evaluation (python package containing necessary implementation for analysis)
        - analysis (IPython Notebooks)
        - data (data used in the analysis)
        - bin (directory with helper bash scripts)
 
 - Project depends on certain python libraries defined in the initialization script
 
    - `./bin/init.sh`
    
 - Unpack the data required for analysis (higgs and converters)
 
    - `./bin/prepare_data.sh`
    
    
### Running the analysis
    
 - Analysis can be reproduced interactively using IPython Notebook
 
    - change the PYTHONPATH in the `bin/start_notebook.sh` to location where code was unzipped
 
    - `./bin/start_notebook.sh`
    
    - open http://localhost:8888 to view existing notebooks
    
    - select `Supervised Classification` notebook and interact with the results
    
    - you can select `Cell->Run All` or execute statements one by one by pressing `Shift+Enter`
    
 - Notebook can subsequently be turned into PDF
 
    - `./bin/convert_notebooks.sh`
    
    - find the `Supervised Classification.pdf` 