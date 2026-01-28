import numpy as np
import pandas as pd
import streamlit as st
import joblib

#lets load all the instances required over here
with open('transformer.joblib', 'rb') as file:
    transformer= joblib.load(file)

#Lets load the model

with open('final_model.joblib', 'rb') as file:
    model= joblib.load(file)

st.title('INN HOTEL GROUP')
st.header('This application will predict the chances of booking cancellation')

#Lets take input from the user
amnth= st.slider('Select your month of arrival', min_value=1, max_value=12)
wkd_lambda= (lambda x: 0 if x=='Mon' 
             else 1 if x=='Tue' 
             else 2 if x=='Wed' 
             else 3 if x=='Thurs' 
             else 4 if x=='Fri' 
             else 5 if x=='Sat' 
             else 6)
awkd=wkd_lambda(st.selectbox('Select your weekday of arrival', ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']))
dwkd=wkd_lambda(st.selectbox('Select your weekday of departure', ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']))
wkend= st.number_input('Enter how many weekend nights are there during stay', min_value=0)
wk=  st.number_input('Enter how many week nights are there during stay', min_value=0)
totn= wkend+wk
mkt= (lambda x: 0 if x=='Offline' else 1)(st.selectbox('How has the booking been made?', ['Offline', 'Online']))
lt= st.number_input('How many day prior the booking was made', min_value=0)
price= st.number_input('What is the avg price per room', min_value=0)
adults= st.number_input('Adult members in booking', min_value=0)
spcl= st.selectbox('Select the number of special requests made:', [0,1,2,3,4,5])
park= (lambda x: 0 if x=='No' else 1)(st.selectbox('Does guest need parking space?',['Yes','No']))


#Transform the data

lt_t, price_t= transformer.transform([[lt, price]])[0]

#Create the input list
input_list= [lt_t, spcl, price_t, adults, wkend, park, wk, mkt, amnth, awkd, totn, dwkd]


#Make prediction
prediction= model.predict_proba([input_list])[:,1][0]


#Lets show probability
if st.button('Predict'):
    st.success(f'Cancellation Chances {(round(prediction,4))*100}%')