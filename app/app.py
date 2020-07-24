from flask import Flask, request, url_for, render_template
from model.predict_model import predict
from model.dataHelperFunctions import skip_headder, select_transform_majors
import numpy as np
import pandas as pd
from io import TextIOWrapper
from joblib import load
from pathlib import Path
import base64

app = Flask(__name__, template_folder='templates')

model = load(Path('model/Model50_15_full_2020-52-03-13-52.z'))

cols = ['sio2', 'tio2', 'al2o3', 'feo_tot', 'mgo', 'cao', 'na2o', 'k2o', 'p2o5']

def get_table_download_link(df):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Lithology_predictions.csv">Download results csv file</a>'
    return href

@app.route('/')
def main():
    return (render_template('index.html'))

@app.route('/single_predict', methods=['POST'])
def single_predict():
    int_values = [x for x in request.form.values()]
    final = np.array(int_values)
    input_df = pd.DataFrame([final], columns= cols)
    output_df = predict(model, input_df)
    output = output_df.iloc[0,:].values.tolist()
    return render_template('index.html', prediction = f'''<div class="alert alert-success">
                                                        Your sample is <strong>{output[1]*100:.2f}% likely to be {output[0]}</strong>
                                                        </div>''')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    file_upload = request.files['file']
    file_upload = TextIOWrapper(file_upload, encoding='utf-8')
    data = select_transform_majors(skip_headder(file_upload))
            
    #TODO extend to include xlxs file input. normalise to 100% anhydrous. recalc Fe2O3

    batch_predict = predict(model, data)

    idx = data.index.to_series(name= 'SampleID')
    idx.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    batch_predict.reset_index(drop=True, inplace=True)
            
    cols_ = {'sio2':'SiO2', 'tio2':'TiO2', 'al2o3':'Al2O3', 'feot':'FeOT', 'mgo':'MgO', 'cao': 'CaO', 'na2o':'Na2O', 'k2o': 'K2O', 'p2o5': 'P2O5'}
    out_df = pd.concat([idx, data, batch_predict],axis=1).rename(columns= cols_)
    download = get_table_download_link(out_df)
    return render_template('index.html',download=download, tables=out_df.to_html(classes=['data','table-striped'], header=True, index=False,border=None))

if __name__== '__main__':
    app.run()