import requests
import json
import streamlit as st
import pandas as pd
import numpy as np

import time
import datetime

# if you want to  create a cell like jupyter notbook then type   the ( # %% ) for the cell '''

# Set page configuration
st.set_page_config(
    page_title="Weather Forcasting Dashboard",
    page_icon="⛅", 
    layout="wide",
    initial_sidebar_state="expanded",
)

api_key = "your api key"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP errors

        data = json.loads(response.text)
        if data['cod'] != 200:
            st.warning(f"Error: {data['message']}!!! " + "Please enter a valid city name")
            return
        
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return
        
    # Extract relevant weather information
    weather_description = data['weather'][0]['description']
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']

    windspeed = data['wind']['speed']
    wind_degree = data['wind']['deg']

    sunrise = data['sys']['sunrise']
    sunset = data['sys']['sunset']
    
    # Convert temperature from Kelvin to Celsius
    temperature = round(temperature - 273.15, 2)
    pressure = round(pressure/1000, 2)
    
    # Print the weather forecast
    st.header(f"Weather in {city}: {weather_description}",divider='rainbow')

    col1, col2 = st.columns([1,3])
    col1.subheader("Temperature in °C🌡")
    col2.subheader(temperature)

    col1.subheader("Humidity in % 💧")
    col2.subheader(humidity)

    col1.subheader("Pressure in atm 🕣")
    col2.subheader(pressure)

    col1.subheader("Wind speed in m/s 💨 ")
    col2.subheader(windspeed)

    col1.subheader("Wind degree 🧭")
    col2.subheader(wind_degree)

    col1.subheader("Sunrise 🌅")
    col2.subheader(datetime.datetime.fromtimestamp(sunrise).strftime('%Y-%m-%d %H:%M:%S'))

    col1.subheader("Sunset 🌇")
    col2.subheader(datetime.datetime.fromtimestamp(sunset).strftime('%Y-%m-%d %H:%M:%S'))

def get_5days_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP errors

        data = json.loads(response.text)
        # if data['cod'] != 200:
        #     st.warning(f"Error: {data['message']}!!! " + "Please enter a valid city name")
        #     return
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return
    
    # Extract relevant weather information
    weather = data['list']
    weather_data = []
    
    for daily in weather:
        date = daily['dt_txt']
        temp = daily['main']['temp'] - 273.15
        humidity = daily['main']['humidity']
        pressure = daily['main']['pressure']
        weather_description = daily['weather'][0]['description']
        wind_speed = daily['wind']['speed']
        wind_degree = daily['wind']['deg']
        weather_data.append([date, temp, humidity, pressure, weather_description, wind_speed, wind_degree])

    weather_df = pd.DataFrame(weather_data, columns=['Date', 'Temperature', 'Humidity', 'Pressure', 'Weather Description', 'Wind Speed', 'Wind Degree'])

    weather_df['Given_City'] = city.title()
    weather_df['Fetched City'] = data['city']['name']
    weather_df['Country'] = data['city']['country']
    weather_df['Population (in Millions)'] = data['city']['population']/1e6

    weather_df = weather_df[['Given_City', 'Fetched City', 'Country', 'Population (in Millions)', 'Date', 'Temperature', 'Humidity', 'Pressure', 'Weather Description', 'Wind Speed', 'Wind Degree']]
    st.header("5 Days Weather Forecast", divider='rainbow')
    st.write(weather_df)
    st.subheader("Ploting the bar graph of variation in Temperature")
    st.bar_chart(weather_df[['Temperature']])

# Add the heading of the Project
st.header(":cloud: Welcome to the Weather Forecasting :sunny:", divider='rainbow')

# Add the image in the project
st.image('./images/weather forecast.jpg', use_column_width = True)
st.subheader("Enter the City/State/Country Name")
city = st.text_input("Enter City Name", "Bhopal")

# Add the button to the project
if st.button("Get Weather Update") and city:
    get_weather(city)
    get_5days_weather(city)


