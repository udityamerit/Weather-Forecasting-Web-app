import requests
import math
import json
import sys
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image





# Set page configuration
st.set_page_config(
    page_title="Weather Forcasting Dasboard",
    page_icon=":cloud:",
    layout="wide",
    initial_sidebar_state="expanded",
)

df = pd.read_csv('Weather forecast data 1.csv')
df1 = pd.read_csv('Weather forecast data set 2.csv')


# st.header('Data set 1:')
# st.dataframe(df)
# st.header('Data set 2:')
# st.dataframe(df1)

# Get the particular data form the specific countery and city 

# x = list(df['name'])
# a = x[0:10]
# st.write(a)

# y = list(df['temp'])
# b = y[0:10]
# st.write(b)

def get_weather(city):
    api_key = "3f4f458fc6d5cb3440d24074d29f7e82"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP errors
    except requests.exceptions.HTTPError as err:
        print(f"Error: {err}")
    
    try:
        data = json.loads(response.text)
        if data['cod'] != 200:
            print(f"Error: {data['message']}")
    except json.JSONDecodeError as err:
        print(f"Error: Failed to parse response JSON - {err}")
        
    # Extract relevant weather information
    weather_description = data['weather'][0]['description']
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    # windspeed = data['main']['wind']['speed']
    
    # Convert temperature from Kelvin to Celsius
    temperature = round(temperature - 273.15, 2)
    
    # Print the weather forecast
    st.header(f"Weather in {city}: {weather_description}",divider='rainbow')

    # st.write(f"Weather in {city}: {weather_description}")
    st.header(f"Temperature 🌡:  {temperature}°C")
    st.header(f"Humidity 💧:  {humidity} %")
    st.header(f"Pressure 🕣:  {pressure} pa")
   
    # st.write(f"Wind speed 🍃🍂:  {windspeed} m/s")

# Add the heading of the Project
st.markdown(":cloud:")
st.header("Welcome to the Weather forecasting", divider='rainbow')

# Add the image in the project
st.image('Weather forecast X.jpg', width= 1400)
st.header("Enter the City/State/Country Name")
city = st.text_input("")

# Add the button to the project
if st.button("Get Weather Update"):
    get_weather(city)

data = pd.DataFrame( np.random.randn(14,3),columns = ['Temperature','Humidity','Pressure'])

# Difine the condition to the project
if city == 'mumbai':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('mumbai.PNG',width= 1400)

elif city == 'bhopal':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('bhopal data 2 week.PNG',width= 950)

elif city == 'tokyo':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('tokyo japan.PNG',width= 950)

elif city == 'london':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('london, uk, england.PNG',width= 950)

elif city == 'jaunpur':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('jaunpur data 2 week.PNG',width= 950)

elif city == 'england':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('london, uk, england.PNG',width= 950)

elif city == 'moscow':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.write(df.head(9))
    st.bar_chart(df[['tempmax','tempmin']].head(9))

    #pyplot chart
    corr = df.corr(df[['tempmax','tempmin']].corr)
    st.write(corr)
    fig, ax = plt.subplot()
    sns.heatmap(corr)
    st.pyplot(fig)
    