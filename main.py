# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 20:48:07 2025

@author: samya
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open('trained_model.pkl', 'rb'))

#Creating function for prediction
def house_prediction(input_data):
    
    input_data_as_numpy=np.asarray(input_data)
    input_data_as_reshaped=input_data_as_numpy.reshape(1,-1)
    prediction=loaded_model.predict(input_data_as_reshaped)
    return (prediction)
    

def main():
    st.title ('House Price Prediction')
    
    MedInc=st.number_input('Enter MedInc')
    HouseAge=st.number_input('Enter house age')
    AveRooms=st.number_input('Enter avg. area no of rooms')
    AveBedrms=st.number_input('Enter avg. area no of bedrooms')
    Population	=st.number_input('Enter area population')
    AveOccup=st.number_input('Enter avg occupancy')
    Latitude=st.number_input('Enter latitude')
    Longitude=st.number_input('Enter longitude')
    Price=''
    if st.button('House price'):
        Price=house_prediction([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude])
    
    st.success(Price)



if __name__=='__main__':
    main()
