#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:06:55 2022

@author: manas
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#Loading the saved model

loaded_model = pickle.load(open('trained_model.sav','rb'))
scaler = pickle.load(open('standardized_data.pkl','rb'))
    

def diabetes_prediction(input_data):

    input_data_as_npArray = np.asarray(input_data)

    input_data_reshaped = input_data_as_npArray.reshape(1,-1)

    std_data = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)

    if(prediction[0]==0):
      return 'The person is non-Diabetic'
    else:
      return 'The person is Diabetic'
  
    

def main():
    
    
    #Title
    
    st.title('Diabetes Prediction System')
    
    #Getting the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin')
    BMI = st.text_input('Body Mass Index')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')
    
    
    #Code for prediction
    
    diagnosis = ''
    
    #Creating a button
    
    if st.button('Diabetes Prediction Result'):
        diagnosis = diabetes_prediction((Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age))
    
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
