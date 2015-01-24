Instructions on how to reproduce the classification analysis
============================================================

### Environment Setup

 - Project root directory: `bag-of-algorithms`
 
    - `cd bag-of-algorithms`
 
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
    
 - Notebook can subsequently be turned into PDF
 
    - `./bin/convert_notebooks.sh`
    
    - find the `Supervised Classification.pdf` 