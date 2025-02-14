import streamlit as st
import requests
import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from streamlit_lottie import st_lottie

# Set page configuration with custom theme
st.set_page_config(
    page_title="Vibrant Weather Analytics Dashboard",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar option to select background color
st.sidebar.title("🎨 Dashboard Settings")
background_color = st.sidebar.color_picker(
    "Choose a Background Color:",
    "#FF9A8B"  # Default color
)

# Apply the selected color to the background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
        background-attachment: fixed;
        background-size: cover;
    }}
    .metric-card {{
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
    }}
    .stButton>button {{
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }}
    .stSelectbox {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }}
    div.stTitle {{
        font-weight: bold;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .streamlit-expanderHeader {{
        background: linear-gradient(90deg, #efd5ff 0%, #515ada 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


API_KEY = "3f4f458fc6d5cb3440d24074d29f7e82"

# Custom color schemes for different weather conditions
WEATHER_COLORS = {
    'Clear': ['#FFD700', '#FFA500'],  # Gold to Orange
    'Clouds': ['#B0C4DE', '#4682B4'],  # Light Steel Blue to Steel Blue
    'Rain': ['#87CEEB', '#4169E1'],    # Sky Blue to Royal Blue
    'Snow': ['#E0FFFF', '#B0E0E6'],    # Light Cyan to Powder Blue
    'Thunderstorm': ['#483D8B', '#191970'],  # Dark Slate Blue to Midnight Blue
    'Default': ['#FF69B4', '#DA70D6']  # Hot Pink to Orchid
}

def get_weather_theme(weather_condition):
    return WEATHER_COLORS.get(weather_condition, WEATHER_COLORS['Default'])

def load_lottie_url(weather_condition):
    animation_urls = {
        'Clear': "https://assets9.lottiefiles.com/packages/lf20_xlky4kvh.json",
        'Clouds': "https://assets8.lottiefiles.com/packages/lf20_KUFdS6.json",
        'Rain': "https://assets1.lottiefiles.com/packages/lf20_rt6mpfp9.json",
        'Snow': "https://assets4.lottiefiles.com/packages/lf20_5i5k8k2f.json",
        'Thunderstorm': "https://assets3.lottiefiles.com/packages/lf20_bb9qzg9h.json"
    }
    
    url = animation_urls.get(weather_condition, animation_urls['Clear'])
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def process_forecast_data(forecast_data):
    """Process the raw forecast data into a pandas DataFrame."""
    data = []
    for item in forecast_data['list']:
        data.append({
            'datetime': datetime.datetime.fromtimestamp(item['dt']),
            'temp': item['main']['temp'],
            'temp_min': item['main']['temp_min'],
            'temp_max': item['main']['temp_max'],
            'humidity': item['main']['humidity'],
            'wind_speed': item['wind']['speed'],
            'wind_deg': item['wind'].get('deg', 0),
            'pressure': item['main']['pressure'],
            'description': item['weather'][0]['description'],
            'main_weather': item['weather'][0]['main']
        })
    return pd.DataFrame(data)

def create_heatmap(lat, lon, temp):
    """Create a heatmap showing temperature for the entered city."""
    # Create a base map centered on the entered city
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles='CartoDB positron')
    
    # Prepare data for the heatmap: [lat, lon, temperature]
    heat_data = [[lat, lon, temp]]
    
    # Add heatmap layer
    HeatMap(heat_data, name="Temperature Heatmap", min_opacity=0.5, max_zoom=18).add_to(m)
    
    # Add a marker for the city
    folium.Marker(
        location=[lat, lon],
        popup=f"Temperature: {temp}°C",
        icon=folium.Icon(color='blue')
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def get_weather_data(city):
    try:
        current_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        
        current_response = requests.get(current_url)
        forecast_response = requests.get(forecast_url)
        
        current_response.raise_for_status()
        forecast_response.raise_for_status()
        
        return current_response.json(), forecast_response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"🌧️ Error fetching weather data: {str(e)}")
        return None, None

def create_colorful_metric(label, value, delta=None, color_scheme=None):
    if color_scheme is None:
        color_scheme = ['#FF69B4', '#DA70D6']
    
    metric_html = f"""
        <div class="metric-card" style="background: linear-gradient(45deg, {color_scheme[0]}, {color_scheme[1]});">
            <h3 style="color: white; margin: 0;">{label}</h3>
            <h2 style="color: white; margin: 10px 0;">{value}</h2>
            {f'<p style="color: white; margin: 0;">Δ {delta}</p>' if delta else ''}
        </div>
    """
    return st.markdown(metric_html, unsafe_allow_html=True)

def main():
    st.sidebar.title("🎨 Dashboard Settings")
    
    popular_cities = {
        "New York 🗽": "New York",
        "London 🇬🇧": "London",
        "Tokyo 🗼": "Tokyo",
        "Paris 🗼": "Paris",
        "Sydney 🦘": "Sydney",
        "Singapore 🇸🇬": "Singapore"
    }
    
    city_option = st.sidebar.radio("Select city input method:", ["Choose from popular cities 🌍", "Enter custom city 🔍"])
    
    if city_option == "Choose from popular cities 🌍":
        city = st.sidebar.selectbox("Select a city:", list(popular_cities.keys()))
        city = popular_cities[city]
    else:
        city = st.sidebar.text_input("Enter city name:", "")
    
    temp_unit = st.sidebar.selectbox("🌡️ Temperature Unit:", ["Celsius", "Fahrenheit"])
    chart_theme = st.sidebar.selectbox("🎨 Chart Theme:", [
        "Vibrant", "Pastel", "Neon", "Ocean", "Sunset"
    ])
    
    chart_colors = {
        "Vibrant": px.colors.qualitative.Set1,
        "Pastel": px.colors.qualitative.Pastel,
        "Neon": px.colors.qualitative.Bold,
        "Ocean": px.colors.sequential.Blues,
        "Sunset": px.colors.sequential.Sunset
    }
    
    st.title("🌈 Vibrant Weather Analytics Dashboard")
    
    if city:
        current_weather, forecast_data = get_weather_data(city)
        
        if current_weather and forecast_data:
            weather_condition = current_weather['weather'][0]['main']
            color_scheme = get_weather_theme(weather_condition)
            
            lottie_weather = load_lottie_url(weather_condition)
            if lottie_weather:
                st_lottie(lottie_weather, height=200, key="weather_animation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_colorful_metric(
                    "Temperature",
                    f"{current_weather['main']['temp']}°C",
                    f"{current_weather['main']['temp_max'] - current_weather['main']['temp_min']:+.1f}°C",
                    color_scheme
                )
            
            with col2:
                create_colorful_metric(
                    "Humidity",
                    f"{current_weather['main']['humidity']}%",
                    "Normal" if 30 <= current_weather['main']['humidity'] <= 70 else "Extreme",
                    color_scheme
                )
            
            with col3:
                create_colorful_metric(
                    "Wind Speed",
                    f"{current_weather['wind']['speed']} m/s",
                    f"Gusts: {current_weather['wind'].get('gust', 0)} m/s",
                    color_scheme
                )
            
            df = process_forecast_data(forecast_data)
            
            # Weather Condition Summary
            st.subheader("🌤️ Weather Condition Summary")
            weather_summary = df['main_weather'].value_counts().reset_index()
            weather_summary.columns = ['Condition', 'Count']
            fig_weather_summary = px.pie(weather_summary, values='Count', names='Condition', title='Weather Condition Distribution', color_discrete_sequence=chart_colors[chart_theme])
            st.plotly_chart(fig_weather_summary, use_container_width=True)
            
            # Hourly Forecast for the next 24 hours
            st.subheader("⏳ Hourly Forecast (Next 24 Hours)")
            hourly_df = df.head(8)  # Next 24 hours (3-hour intervals)
            fig_hourly = px.line(hourly_df, x='datetime', y='temp', title='Hourly Temperature Forecast', color_discrete_sequence=chart_colors[chart_theme])
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Temperature Forecast (Line Chart)
            st.subheader("🌡️ Temperature Forecast")
            fig_temp = px.line(df, x='datetime', y='temp', title='Temperature Forecast', color_discrete_sequence=chart_colors[chart_theme])
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Humidity Forecast (Area Chart)
            st.subheader("💧 Humidity Forecast")
            fig_humidity = px.area(df, x='datetime', y='humidity', title='Humidity Forecast', color_discrete_sequence=chart_colors[chart_theme])
            st.plotly_chart(fig_humidity, use_container_width=True)
            
            # Wind Speed Distribution (Bar Chart)
            st.subheader("🌬️ Wind Speed Distribution")
            fig_wind = px.bar(df, x='datetime', y='wind_speed', title='Wind Speed Distribution', color_discrete_sequence=chart_colors[chart_theme])
            st.plotly_chart(fig_wind, use_container_width=True)
            
            # Pressure Trends (Scatter Plot)
            st.subheader("📊 Pressure Trends")
            fig_pressure = px.scatter(df, x='datetime', y='pressure', title='Pressure Trends', color_discrete_sequence=chart_colors[chart_theme])
            st.plotly_chart(fig_pressure, use_container_width=True)
            
             # Heatmap for the entered city
            st.subheader("🔥 Temperature Heatmap for the Entered City")
            lat = current_weather['coord']['lat']
            lon = current_weather['coord']['lon']
            temp = current_weather['main']['temp']
            heatmap = create_heatmap(lat, lon, temp)
            folium_static(heatmap, width=1200, height=600)
            
               

if __name__ == "__main__":
    main()