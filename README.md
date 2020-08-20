Rock protolith prediction app 
==============================

ML model trained to predict if a rock is igneous or sedimentary based on major element geochemistry.

This model is provided as a web app which can be found at: http://protolith-app.ap-southeast-2.elasticbeanstalk.com/

To run locally
------------
* Setup a virtualenv using conda or venv running Python 3.6.10
* `pip3 install -r requirements.txt` 
* Make a new directory and copy/clone app folder
* In the app directory run
`flask run` (may need to move the wsgi.py file out of app folder)

Or with Docker 
* In app folder run
* `docker build -t protolith-app .`
* `docker run -dp 8000:8000 protolith-app`
* The app will then be available via web browser at localhost:8000  

Attribution and training data
------------
This predictor is a reformulation of the original work published by Hasterok et al 2019. 
*Chemical identification of metamorphic protoliths using machine learning methods*. Computers and Geosciences. **132**, 56-68

The source data for training can be found at: https://zenodo.org/record/2586461#.XvwL8cgzZPY

Project Organization
------------

    ├── README.md 
    ├── notebooks
    │   ├── model_assessment.ipynb      
    │   └── train_protolith_model.ipynb
    └── app               
        ├── requirements.txt   
        ├── templates           
        │   └── index.html
        ├── static           
        ├── model         <- App helper functions and final trained model
        │   │                 
        │   ├── dataHelperFunctions.py 
        |   ├── predict_model.py
        │   └── model
        ├── Dockerfile
        ├── wsgi.py <- Gunicorn wsgi server
        └── app.py  <- Flask protolith predictor app script
           
    
--------


