
"""
Protolith classification model using major element geochemistry and a balanced random forrest algorithm to discriminate sedimentary from ignous protoliths. 

Scripts to create and train final model pipeline on complete dataset.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from imblearn.ensemble import BalancedRandomForestClassifier

import joblib

from datetime import datetime
from pathlib import Path

in_path = r'D:/Python ML/SA_Geology_protolith_predictions/data/processed/classifier_data_2019-02-26_combined.csv'
out_path = r'D:/Python ML/SA_Geology_protolith_predictions/models'

def load_training_data(path: str) -> pd.DataFrame:
    """
    Function to load the protolith model training data from file. Data is already in Atchison simplex form.
    Parameters:
        path: path to training data as string
    returns:
        training dataset X_train, y_train
    """
    p = Path(path)
    try:
        if p.exists and p.suffix == '.csv':
            df = pd.read_csv(p) #training set
    except:
        print('could not read training data file')

    #create raw X training set and binary labeled y array
    X_train, y_train = df[['SiO2','TiO2','Al2O3','FeOT','MgO','CaO','Na2O','K2O','P2O5']], df[['ROCK_GROUP']].replace(['igneous','sedimentary'],[0,1])

    return X_train, y_train  

def train_model(X: pd.DataFrame, y: pd.DataFrame) -> sklearn.pipeline.Pipeline:
    """
    Function to train the protolith classification model.
    Paramaters:
        X: training data, default X_train
        y: training labels, default y_train
    returns
        trained model pipeline
    """
    # Create a pipeline to scale and train a calibrated balanced random forrest classifier
    cv = StratifiedKFold(n_splits=5, random_state=101)

    clsf_pipe = Pipeline([('sc', StandardScaler()),
                ('classifier', CalibratedClassifierCV(base_estimator = BalancedRandomForestClassifier(n_estimators= 1000, max_depth=50, min_samples_leaf= 1, min_samples_split= 2,
                max_features= 'sqrt',sampling_strategy='not minority', n_jobs= -1, random_state= 101), method= 'sigmoid', cv=cv))])
    # fit the model
    model_pipe = clsf_pipe.fit(X, y)
    
    return model_pipe

def save_model(model, name: str, out_path: str):
    """
    Function to save trained protolith classification model.
    Paramaters:
        model: trained sklearn model pipe
        name: str. File name (will be appended with date and time)
        out_path: str. path to export file to
    returns
        joblib file
    """
    now = datetime.now().strftime("%Y-%M-%d-%H-%M")
    file_name = f'{name}_{now}.joblib' 
    p = Path(out_path) / file_name
    joblib.dump(model, p)

