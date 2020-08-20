import numpy as np
import pandas as pd

pd.set_option('precision', 3)

def predict(model, input_df):
    """
    Function to predict results from trained protolith classifier model.
    
    Parameters:
    -------------------
    model: trained classifier model pipeline
    input_df: pandas.DataFrame with 9 element major chemistry data in Atchison symplex
    
    Returns:
    ------------------
    pandas.DataFrame containing predicted class and probability
    """
    _class = pd.DataFrame(model.predict(input_df))
    _prob = pd.DataFrame(model.predict_proba(input_df))
    predictions = pd.concat([_class, _prob], axis=1)
    predictions.columns = ['Class', 'Proba_0', 'Proba_1']
    predictions['Probability'] = np.where((predictions.Class == 0), predictions.Proba_0, predictions.Proba_1)
    predictions['Class'] = predictions['Class'].replace([0,1],['igneous','sedimentary'])
    return predictions.drop(['Proba_0','Proba_1'],axis=1)

def get_output_df(data, predictions):
    """
    Function to take transformed input data and resultant predictions from 
    model and generate formatted out put DataFrame.
    
    Parameters:
    --------------
    data: pandas.DataFrame transformed major element data 
    predictions: pandas.DataFrame output from predict function
    
    Returns:
    --------------
    pandas.DataFrame with input data and generated predictions.
    """
    idx = data.index.to_series(name= 'SampleID')
    idx.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    predictions.reset_index(drop=True, inplace=True)
            
    cols_ = {'sio2':'SiO2', 'tio2':'TiO2', 'al2o3':'Al2O3', 'feot':'FeOT', 'mgo':'MgO', 'cao': 'CaO', 'na2o':'Na2O', 'k2o': 'K2O', 'p2o5': 'P2O5'}
    out_df = pd.concat([idx, data, predictions],axis=1).rename(columns= cols_)
    return out_df
