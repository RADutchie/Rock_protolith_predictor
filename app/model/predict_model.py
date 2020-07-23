import numpy as np
import pandas as pd


pd.set_option('precision', 3)

def predict(model, input_df):
    """
    Function to predict results from trained protolith classifier model.
    Inputs:
        model = trained classifier model pipeline
        input_df = pandas dataframe with 9 element major chemistry data in Atchison symplex
    Returns:
        Pandas df containing prediceted class and probability
    """
    _class = pd.DataFrame(model.predict(input_df))
    _prob = pd.DataFrame(model.predict_proba(input_df))
    predictions = pd.concat([_class, _prob], axis=1)
    predictions.columns = ['Class', 'Proba_0', 'Proba_1']
    predictions['Probability'] = np.where((predictions.Class == 0), predictions.Proba_0, predictions.Proba_1)
    predictions['Class'] = predictions['Class'].replace([0,1],['igneous','sedimentary'])
    return predictions.drop(['Proba_0','Proba_1'],axis=1)


