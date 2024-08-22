import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
import pygwalker as pyg
import datetime
from pygwalker.api.streamlit import StreamlitRenderer
import os

# Set page configuration
st.set_page_config(
    page_title="Weather Forecasting Dashboard",
    page_icon="⛅", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# api_key = "your api key"

headers = {
    "authorization": st.secrets["auth_var"],
    "content-type":"application/json"
}
@st.cache_data
def get_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    data = json.loads(response.text)
    return data

@st.cache_data
def get_5days_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    data = json.loads(response.text)
    return data

def display_weather(city):
    data = get_weather_data(city)
    if data['cod'] != 200:
        st.warning(f"Error: {data['message']}!!! " + "Please enter a valid city name")
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
    st.header(f"Weather in {city}: {weather_description}", divider='rainbow')
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

def display_5days_weather(city):
    data = get_5days_weather_data(city)
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
    file_path = 'weather.csv'
    weather_df.to_csv(file_path, index=False)
    
    st.subheader("Plotting the bar graph of variation in Temperature")
    st.bar_chart(weather_df[['Temperature']], color='#ffaa0088')
    
    return weather_df

# Add the heading of the Project
st.header(":cloud: Welcome to the Weather Forecasting :sunny:", divider='rainbow')

# Use the absolute path of the image file
# image_path = os.path.abspath('./images/weather_forecast.jpg')
st.image('./images/weather forecast.jpg', use_column_width = True)
st.subheader("Enter the City/State/Country Name")


city = st.text_input("Enter City Name", "Bhopal")

# Add the button to the project
if st.button("Get Weather Update") and city:
    display_weather(city)
    weather_df = display_5days_weather(city)
    
    # Display the Pygwalker explorer outside the function to retain state

uploaded_file =  st.file_uploader("Your csv data ")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    pyg_aap = StreamlitRenderer(df)
    pyg_aap.explorer()
