import requests
import math
import json
import sys
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# if you want to  create a cell like jupyter notbook then type   the ( # %% ) for the cell '''


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
    df2 = pd.DataFrame({'lat':[19.082],
                        'lon':[72.8789]})
    st.map(df2, size=200, color='#0044ff')
    data = pd.DataFrame(
        np.random.randn(10,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'bhopal':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('bhopal.png',width= 1400)
    df2 = pd.DataFrame({'lat':[23.264],
                        'lon':[77.402]})
    st.map(df2, size=200, color='#FF5733')
    data = pd.DataFrame(
        np.random.randn(15,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'tokyo':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('tokyo japan.PNG',width= 1400)
    df2 = pd.DataFrame({'lat':[35.6764],
                        'lon':[139.650]})
    st.map(df2, size=200, color='#FF5733')
    data = pd.DataFrame(
        np.random.randn(14,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'london':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('london, uk, england.PNG',width= 1400)
    df2 = pd.DataFrame({'lat':[51.51279],
                        'lon':[-0.09184]})
    st.map(df2, size=200, color='#FF5733')
    data = pd.DataFrame(
        np.random.randn(19,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'fatehpur':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('fatehpur.png',width= 1400)
    df2 = pd.DataFrame({'lat':[25.9210],
                        'lon':[80.7996]})
    st.map(df2, size=200, color='#FF5733')
    data = pd.DataFrame(
        np.random.randn(35,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data) 

elif city == 'kanpur':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('kanpurr.png',width= 1400)
    df2 = pd.DataFrame({'lat':[26.464],
                        'lon':[80.32]})
    st.map(df2, size=200, color='#56F50C')
    data = pd.DataFrame(
        np.random.randn(25,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'allahabad':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('prayag.png',width= 1400)
    df2 = pd.DataFrame({'lat':[25.441],
                        'lon':[81.835]})
    st.map(df2, size=200, color='#F9FF33')
    data = pd.DataFrame(
        np.random.randn(14,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'sehore':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('sehore.png',width= 1400)
    df2 = pd.DataFrame({'lat':[23.2082],
                        'lon':[77.0844]})
    st.map(df2, size=200, color='#0CF5E0')
    data = pd.DataFrame(
        np.random.randn(12,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'usa':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('usa.png',width= 1400)
    df2 = pd.DataFrame({'lat':[40.3588],
                        'lon':[-103.1743]})

    st.map(df2, size=200, color='#E70CF5')
    data = pd.DataFrame(
        np.random.randn(14,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'jaunpur':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('jaunpur.png',width= 1400)
    df2 = pd.DataFrame({'lat':[25.744],
                        'lon':[82.6837]})
    st.map(df2, size=200, color='#0044ff')
    data = pd.DataFrame(
        np.random.randn(10,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'england':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('england.png',width= 1400)
    df2 = pd.DataFrame({'lat':[52.3555],
                        'lon':[1.1743]})
    st.map(df2, size=200, color='#F50C56')
    data = pd.DataFrame(
        np.random.randn(15,3),
        columns =['Temperature','Humidity','Pressure',])
    st.header("""Graph of temp vs humidity vs pressure """)
    st.line_chart(data)

elif city == 'moscow':
    st.header(" 2 Week Extended forecasting of weather 👇", divider='rainbow')
    st.image('russia.png', width=1400)
    st.write(df.head(19))
    st.area_chart(df[['tempmax','tempmin']].head(9))
    df2 = pd.DataFrame({'lat':[55.755],
                        'lon':[37.617]})
    st.map(df2, size=200, color='#F50C0C')
    

   
