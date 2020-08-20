import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from joblib import load
from pathlib import Path

TAS_model = load(Path('model/TAS.modelfields'))
sand_model = load(Path('model/sandclass.modelfields'))

def litho_classification_cols(df):
    """
    Takes processed dataframe and returns dataframe with required columns.

    Parameters
    ------------
    df: pandas.DataFrame

    Returns
    ------------
    df: pandad.Dataframe
    """
    df["Fe2O3"] = df["FeOT"]*1.11111
    df["totalalkali"] = df["Na2O"] + df["K2O"]
    df["logSiAl"] = np.log10((df['SiO2']/df['Al2O3']))
    df["logFeK"] = np.log10((df["Fe2O3"]/df['K2O']))

    return df


def predict_lithology(model, df, cols):
    """
    Function to assign a lithology classifier based on an extended TAS classification for igneous rocks.
    Parameters:
    --------------
    model: deterministic model fields
    df: pandas.DataFrame
    cols:  [list of cols for classification] 
    Returns:
    -------------
    pandas.DataFrame with appended igneous lithology classification
    """
    fclasses = [k for (k, v) in model.items()]
    points = df.loc[:, cols].values
    
    polys = [Polygon(model[c]["poly"], closed= True) for c in fclasses]
    name = [model[c]["names"] for c in fclasses]

    indexes = np.array([p.contains_points(points) for p in polys]).T
    notfound = np.logical_not(indexes.sum(axis=-1))
    out = pd.Series(index=df.index)
    outlist = list(map(lambda ix: name[ix], np.argmax(indexes, axis=-1)))
    out.loc[:] = outlist
    out.loc[(notfound)] = "unclassified"
    return out

def classify_lithology(df):
    """
    Function to generate a lithology column based on a deterministic classifier and chemical composition
    Parameters:
    --------------
    df: pandas.DataFrame with protolith predictions and major element data
    Returns:
    --------------
    pandas.DataFrame with added lithology classifier column
    """

    df1 = litho_classification_cols(df)
    tas = ["SiO2", "totalalkali"]
    snd = ["logSiAl", "logFeK"]

    df1["TAS"] = predict_lithology(TAS_model, df1, tas)
    df1["sand"] = predict_lithology(sand_model, df1, snd)
    df1['Lithology'] = np.where((df1.Class == 'igneous'), df1.TAS, 
                    (np.where((df1.MgO >= 10), ['Dolomite'], 
                    (np.where((df1.CaO >= 15), ['Limestone'], df1.sand)))))
    return df1.drop(["totalalkali", "logSiAl", "logFeK", "TAS", "sand", "Fe2O3"], axis=1)