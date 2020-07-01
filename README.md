Rock protolith prediction app 
==============================

ML model trained to predict missing lithology names based on major element geochemistry.

This model is provided as a streamlit based web app which can be found at: TODO make live app

To run localy
------------
* Setup a virtualenv using conda or venv running Python 3.6.10
  `pip3 install -r requirements.txt` 
* Make a new directory and copy/clone src folder
* In the src directory run
`streamlit run protolith_app.py`

Attribution and training data
------------
This predictor is a reformulation of the origional work published by Hasterok et al 2019. 
*Chemical identification of metamorphic protoliths using machine learning methods*. Computers and Geosciences. **132**, 56-68

The source data for training can be found at: https://zenodo.org/record/2586461#.XvwL8cgzZPY

Project Organization
------------

    ├── README.md          
    ├── notebooks          <- Model development and training notebooks
    │    ├── 0.3-dutch-ensemble-model-assesment                     
    │    ├── 0.4-dutch-tune-chosen-model                     
    │    └── EDA-dutch-Hasterok-training-data   
    |
    ├── reports            <- Model experiment results training data stats
    │   └── figures        <- Final trained model metrics
    │
    ├── requirements.txt   
    │
    └── src                
        ├── __init__.py    
        │
        ├── data           <- Script to download or generate data
        │   └── make_dataset.py
        │
        ├── images         <- images for app  
        ├── models         <- Scripts to train models, make predictions, app helper funcions and
        │   │                 final trained model
        │   ├── dataHelperFunctions.py 
        |   ├── predict_model.py
        │   └── trainProtolithModel.py
        |   
        │
        └── protolith_app.py  <- Streamlit protolith predictor app script
           
    
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
"# Rock_protolith_predictor" 
