import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from streamlit_lottie import st_lottie
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set page config
st.set_page_config(
    page_title="AI-Powered Weather & AQI Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Key for OpenWeatherMap
API_KEY = "3f4f458fc6d5cb3440d24074d29f7e82"
AQI_API_KEY = "89f38a329e46ba8ddc6896d909b9db96e57d81a1"

# Function to fetch weather data
def get_weather_data(city):
    try:
        current_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        
        current_response = requests.get(current_url).json()
        forecast_response = requests.get(forecast_url).json()
        
        return current_response, forecast_response
    except:
        st.error("Failed to fetch weather data.")
        return None, None

# Function to fetch AQI data
def get_aqi_data(city):
    try:
        aqi_url = f"http://api.waqi.info/feed/{city}/?token={AQI_API_KEY}"
        aqi_response = requests.get(aqi_url).json()
        if isinstance(aqi_response, dict) and 'data' in aqi_response:
            return aqi_response
        else:
            st.error("Invalid AQI data received.")
            return None
    except:
        st.error("Failed to fetch AQI data.")
        return None

# Preprocess Forecast Data
def process_forecast_data(forecast_data):
    if not isinstance(forecast_data, dict) or 'list' not in forecast_data:
        st.error("❌ API response does not contain forecast data. Check your city name or API key.")
        return pd.DataFrame()

    data = []
    for item in forecast_data['list']:
        data.append({
            'datetime': datetime.datetime.fromtimestamp(item['dt']),
            'temp': item['main']['temp'],
            'humidity': item['main']['humidity'],
            'wind_speed': item['wind']['speed']
        })
    return pd.DataFrame(data)

# Main Dashboard UI
def main():
    st.title("🌦️ AI-Powered Weather & AQI Dashboard")
    city = st.sidebar.text_input("Enter city name:", "New York")
    if city:
        current_weather, forecast_data = get_weather_data(city)
        aqi_data = get_aqi_data(city)
        
        if current_weather and forecast_data:
            df = process_forecast_data(forecast_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temperature", f"{current_weather['main']['temp']}°C")
            with col2:
                st.metric("Humidity", f"{current_weather['main']['humidity']}%")
            with col3:
                st.metric("Wind Speed", f"{current_weather['wind']['speed']} m/s")
            
            if aqi_data:
                aqi_value = aqi_data.get('data', {}).get('aqi', 'N/A')
                st.subheader("🌍 Air Quality Index (AQI)")
                st.metric("AQI Level", f"{aqi_value}")
            
            st.subheader("📊 Weather Forecast")
            fig = px.line(df, x='datetime', y='temp', title='Temperature Trend')
            st.plotly_chart(fig)
            
            st.subheader("🔥 Temperature Heatmap")
            heatmap = folium.Map(location=[current_weather['coord']['lat'], current_weather['coord']['lon']], zoom_start=10)
            HeatMap([[current_weather['coord']['lat'], current_weather['coord']['lon'], current_weather['main']['temp']]]).add_to(heatmap)
            folium_static(heatmap)
            
if __name__ == "__main__":
    main()
