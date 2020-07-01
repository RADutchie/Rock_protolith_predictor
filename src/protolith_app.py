from models.predict_model import predict
from models.dataHelperFunctions import skip_headder, select_transform_majors
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import base64

# load trained model pipeline
model = joblib.load(Path('models/brf_clf_tuned_cal_pipe_20200611_joblib.z'))

def get_table_download_link(df):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Lithology_predictions.csv">Download results csv file</a>'
    return href

def run():

    st.title("Rock Protolith Prediction App")
    from PIL import Image
    image = Image.open(Path('images/DSCN6550.JPG'))

    st.image(image,use_column_width= True)
   
    #st.sidebar.title('Welcome')

    
    st.markdown("This app tries to predict a rocks protolith from major element geochemistry using an algorithm trained on over half a million labeled global geochemical data.")
    st.header('How to use')
    st.markdown("""Select if you'd like to enter a single sample via the app or upload a csv containing multiple samples. 
    Data should be in weight% oxides and must contain the below elements.""")

    add_selectbox = st.selectbox("Select How would you like to predict?", ("batch", "single sample"))

    st.markdown("Example minimum data required in input csv, including sample identifier in column 1 and headder row. Currently all Fe must be provided as total FeO")
    eg_df = pd.DataFrame(data={'SampleID': [873827], 'SiO2':[73.434], 'TiO2':[0.410], 'Al2O3':[14.488], 'FeOT':[0.554], 'MgO':[0.441], 'CaO': [1.723], 'Na2O':[3.240], 'K2O': [5.588], 'P2O5': [0.123]})
    st.table(eg_df,)

    if add_selectbox == 'single sample':
        st.sidebar.info("Enter your major element oxide data below as wt%, then hit the $predict$ button. Data will be normalised to 100% anhydrous")
        sio2 = st.sidebar.number_input('SiO2')
        tio2 = st.sidebar.number_input('TiO2')
        al2o3 = st.sidebar.number_input('Al2O3')
        feo_tot = st.sidebar.number_input('FeOT')
        mgo = st.sidebar.number_input('MgO')
        cao = st.sidebar.number_input('CaO')
        na2o = st.sidebar.number_input('Na2O')
        k2o = st.sidebar.number_input('K2O')
        p2o5 = st.sidebar.number_input('P2O5')
        #TODO add LOI's, Fe2O3


        input_dict = {'sio2':sio2, 'tio2':tio2, 'al2o3':al2o3, 'feo_tot':feo_tot, 'mgo':mgo, 'cao': cao, 'na2o':na2o, 'k2o': k2o, 'p2o5': p2o5}
        #TODO need to normalise to 100% anhydrous, recalc Fe2O3
        input_df = pd.DataFrame([input_dict])

        if st.button('Predict'):
            output_df = predict(model, input_df)
            output = output_df.iloc[0,:].values.tolist()
            st.success(f'Your sample is {output[1]*100:.2f}% likely to be {output[0]}')
            
        
    if add_selectbox == 'batch':
        file_upload = st.file_uploader("Upload csv file for batch predictions", type=['csv'])

        if file_upload is not None:
            data = select_transform_majors(skip_headder(file_upload))
            
            #TODO extend to include xlxs file input. normalise to 100% anhydrous. recalc Fe2O3

            batch_predict = predict(model, data)

            idx = data.index.to_series(name= 'SampleID')
            idx.reset_index(drop=True, inplace=True)
            data.reset_index(drop=True, inplace=True)
            batch_predict.reset_index(drop=True, inplace=True)
            
            cols = {'sio2':'SiO2', 'tio2':'TiO2', 'al2o3':'Al2O3', 'feot':'FeOT', 'mgo':'MgO', 'cao': 'CaO', 'na2o':'Na2O', 'k2o': 'K2O', 'p2o5': 'P2O5'}
            out_df = pd.concat([idx, data, batch_predict],axis=1).rename(columns= cols)
            
            
            st.markdown(get_table_download_link(out_df), unsafe_allow_html=True)
            st.table(out_df)
            
            

    st.markdown('### Contact, references and source code')
    st.markdown('Contact me @RADutchie on twitter or GitHub for comments or issues')
    st.markdown('https://github.com/RADutchie/Rock_protolith_predictor for model and source code')
    st.markdown("""This predictor is a reformulation of the origional work published by Hasterok et al 2019. 
    *Chemical identification of metamorphic protoliths using machine learning methods*. Computers and Geosciences. **132**, 56-68""")

if __name__ == '__main__':
    run()
