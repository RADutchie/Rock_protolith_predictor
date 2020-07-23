import re
import pandas as pd


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
    if re.match(r'^feo(.*?)$', col):
        return 'feot'
    else:
        return col


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
      
    df = df.rename(columns=rename_FeO)

    df = df[['sio2', 'tio2', 'al2o3', 'feot', 'mgo', 'cao', 'na2o', 'k2o', 'p2o5']] # select required elements

    return df.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna() # convert non numeric to Nan and drop rows with missing values