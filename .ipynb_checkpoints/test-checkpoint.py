import sys, argparse
import pandas as pd
import numpy as np
from sklearn.externals import joblib

allowed_descriptions = ['CARDIOVASCULAR','DIGESTIVE SYSTEM','EAR','ENDOCRINE SYSTEM','EYE','FEMALE GENITALIA','INTEGUMENTARY','LYMPHATIC','MALE GENITALIA','MATERNITY','MEDIASTINUM','MUSCULOSKELETAL','RESPIRATORY','URINARY']

ss = joblib.load('ss.joblib')
pca = joblib.load('pca.joblib')
knn = joblib.load('knn.joblib')
autoencoder = joblib.load('autoencoder.joblib')

def classify(**kwargs):
    description = kwargs['description']
    allowed_charges = float(kwargs['allowed_charges'])
    allowed_services = int(kwargs['allowed_services']) 
    payment = float(kwargs['payment']) 
    
    if description not in allowed_descriptions:
        print('Description not valid')
        return

    average_allowed_charges = allowed_charges / allowed_services
    average_payment = payment / allowed_services


    is_cardiovascular = 1 if description == 'CARDIOVASCULAR' else 0
    is_digestive = 1 if description == 'DIGESTIVE SYSTEM' else 0
    is_ear = 1 if description == 'EAR' else 0
    is_endocrine = 1 if description == 'ENDOCRINE SYSTEM' else 0
    is_eye = 1 if description == 'EYE' else 0
    is_female = 1 if description == 'FEMALE GENITALIA' else 0
    is_inte = 1 if description == 'INTEGUMENTARY' else 0
    is_lymp = 1 if description == 'LYMPHATIC' else 0
    is_male = 1 if description == 'MALE GENITALIA' else 0
    is_maternity = 1 if description == 'MATERNITY' else 0
    is_mediastinum = 1 if description == 'MEDIASTINUM' else 0
    is_muscu = 1 if description == 'MUSCULOSKELETAL' else 0
    is_resp = 1 if description == 'RESPIRATORY' else 0
    is_uri = 1 if description == 'URINARY' else 0

    data = np.array([allowed_charges, allowed_services, average_allowed_charges, average_payment, is_cardiovascular, is_digestive, is_ear, is_endocrine, is_eye, is_female, is_inte, is_lymp, is_male, is_maternity, is_mediastinum, is_muscu, payment, is_resp, is_uri])

    data_std = ss.transform(data.reshape(1, -1))
    data_pca = pca.transform(data_std)
    knn_y = knn.predict(data_pca)
    autoencoder_y = autoencoder.predict(data_pca)

    print('KNN:', knn_y)
    print('AutoEncoder:', autoencoder_y)


    
parser=argparse.ArgumentParser()

parser.add_argument('--description')
parser.add_argument('--allowed_charges')
parser.add_argument('--allowed_services')
parser.add_argument('--payment')

if __name__=='__main__':
    args=parser.parse_args()
    classify(**vars(args))