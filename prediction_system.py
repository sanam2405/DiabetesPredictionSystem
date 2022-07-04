# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#Loading the saved model

loaded_model = pickle.load(open('/Users/manas/Desktop/MachineLearning/DiabetesPrediction/trained_model.sav','rb'))
scaler = pickle.load(open('/Users/manas/Desktop/MachineLearning/DiabetesPrediction/standardized_data.pkl','rb'))
                           
input_data = (1,85,66,29,0,26.6,0.351,31) #Non_Diabetic Data
# input_data = (6,148,72,35,0,33.6,0.627,50) #Diabetic Data

input_data_as_npArray = np.asarray(input_data)

input_data_reshaped = input_data_as_npArray.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)

prediction = loaded_model.predict(std_data)

if(prediction[0]==0):
  print('The person is non-Diabetic')
else:
  print('The person is Diabetic')