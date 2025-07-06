# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 17:01:23 2025

@author: samyayan
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open('trained_model.sav', 'rb'))

#Creating function for prediction
def house_prediction(input_data):
    
    input_data_as_numpy=np.asarray(input_data)
    input_data_as_reshaped=input_data_as_numpy.reshape(1,-1)
    prediction=loaded_model.predict(input_data_as_reshaped)
    return (prediction)
    

def main():
    st.title ('House Price Prediction')
    
    avg_area_income=st.number_input('Enter avg. area income')
    avg_area_house_age=st.number_input('Enter avg. area house age')
    avg_area_number_of_rooms=st.number_input('Enter avg. area no of rooms')
    avg_area_number_of_bedrooms=st.number_input('Enter avg. area no of bedrooms')
    area_population=st.number_input('Enter area population')
    
    Price=''
    if st.button('House price'):
        Price=house_prediction([avg_area_income, avg_area_house_age, avg_area_number_of_rooms, avg_area_number_of_bedrooms, area_population])
    
    st.success(Price)



if __name__=='__main__':
    main()