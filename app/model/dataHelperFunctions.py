import re
import pandas as pd
import numpy as np


def skip_headder(input_file,**kwargs):
    """
    function to parse headder of input csv file and skip to chemical data.
    
    Parameters
    ----------------
    input_file : StringIO object : uploaded input file

    Returns
    ----------------
    pandas data frame
    """
    f = input_file 
        
    match = (r'SiO2|sio2') #regex match to part of required line
    pos = 0
    cur_line = f.readline()
    while re.search(match, cur_line) == None: #iterates line by line till match found
        pos = f.tell()
        cur_line = f.readline()
    f.seek(pos)
        
    return pd.read_csv(f, index_col= 0, **kwargs) 

def rename_FeO(col):
    """
    Function to find variations of the FeO column headder and convert to 'feot'
    """
    if re.match(r'^feo(.*?)$', col):
        return 'feot'
    else:
        return col

def renormalise(df: pd.DataFrame):
    """
    Renormalises compositional data to ensure closure.

    A big thanks to the pyrolite library for this source code: https://github.com/morganjwilliams/pyrolite

    Parameters
    ------------
    df : :class:`pandas.DataFrame`
        Dataframe to renomalise.
    
    Returns
    --------
    :class:`pandas.DataFrame`
        Renormalized dataframe.
    """
    dfc = df.copy(deep=True)
    dfc = dfc.divide(dfc.sum(axis=1).replace(0, 100.0), axis=0) * 100.0
    return dfc

def Fe2O3_variants(df):
    """
    Function to search column index to find Fe2O3 naming variants

    Parameters
    ---------
    cols : df.columns :class:`pandas.Dataframe`
    
    Returns
    ---------
    str : matched regex search for Fe2O3 variants
    """        
    for x in list(df.columns):
        match = re.match(r'^fe2o3(.*?)$', x)
        if match is not None:
            return match.group(0)

def convert_to_FeO(df: pd.DataFrame):
    """
    Converts Fe2O3 total or combination of FeO and Fe2O3 to all FeO total 
    and drops original values from df

    A big thanks to the pyrolite library for how to make this work: https://github.com/morganjwilliams/pyrolite

    Parameters
    -------------
    df : :class:`pandas.DataFrame`
    
    Returns
    -------------
    df : :class:`pandas.DataFrame`
        dataframe with Fe converted to FeO total and original Fe cols removed
    """    
    match = Fe2O3_variants(df) #call function to get Fe2O3 naming variants
    
    if 'feo' in df.columns and 'fe2o3' in df.columns:
        df.fillna({'feo':0.0, 'fe2o3':0.0}, inplace=True)
        df['feototal'] = df['fe2o3']*0.899 + df['feo']
        return df.drop(['fe2o3','feo'], axis=1)

    elif 'feo' not in df.columns and match in df.columns:
        df['feototal'] = df[match]*0.899
        return df.drop([match], axis=1)
    
    else:
        return df    

def select_transform_majors(df):
    """
    Selects only required major element data and formats for model pipeline

    Parameters
    --------------
    in_df: pandas.DataFrame

    Returns
    --------------
    df: pandas.DataFrame
    """
    df.columns = map(str.lower, df.columns) #lowercase column names
    
    df = convert_to_FeO(df)
      
    df = df.rename(columns=rename_FeO)

    df = df[['sio2', 'tio2', 'al2o3', 'feot', 'mgo', 'cao', 'na2o', 'k2o', 'p2o5']] # select required elements

    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna() # convert non numeric to Nan and drop rows with missing values

    return renormalise(df)

