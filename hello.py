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
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import json
import googletrans
from googletrans import Translator

# Set page configuration with custom theme
st.set_page_config(
    page_title="ATMOSVISION - Multi-Page Weather Analytics",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
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
            'main_weather': item['weather'][0]['main'],
            'feels_like': item['main']['feels_like'],
            'uv_index': item.get('uvi', 5),  # Default UV index if not provided
            'visibility': item.get('visibility', 10000),  # Default visibility in meters
            'rain_prob': item.get('pop', 0) * 100 if 'pop' in item else 0  # Probability of precipitation
        })
    return pd.DataFrame(data)

def create_heatmap(lat, lon, temp):
    """Create a heatmap showing temperature for the entered city."""
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles='CartoDB positron')
    heat_data = [[lat, lon, temp]]
    HeatMap(heat_data, name="Temperature Heatmap", min_opacity=0.5, max_zoom=18).add_to(m)
    folium.Marker(
        location=[lat, lon],
        popup=f"Temperature: {temp}°C",
        icon=folium.Icon(color='blue')
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m

def create_city_area_map(lat, lon, city_name):
    """Create a map showing the city's approximate area."""
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles='CartoDB positron')
    
    delta = 0.05  # Approx 5-10 km radius depending on city size
    city_boundary = [
        [lat - delta, lon - delta],
        [lat - delta, lon + delta],
        [lat + delta, lon + delta],
        [lat + delta, lon - delta]
    ]
    
    folium.Polygon(
        locations=city_boundary,
        color='blue',
        weight=2,
        fill=True,
        fill_color='blue',
        fill_opacity=0.2,
        popup=f"{city_name} Area"
    ).add_to(m)
    
    folium.Marker(
        location=[lat, lon],
        popup=city_name,
        icon=folium.Icon(color='blue')
    ).add_to(m)
    
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

# NEW FUNCTION: Get historical weather data for the past 7 days
def get_historical_weather(lat, lon):
    try:
        # Calculate timestamps for the past 7 days
        end_date = datetime.datetime.now()
        dates = [(end_date - datetime.timedelta(days=i)) for i in range(1, 8)]
        
        historical_data = []
        
        for date in dates:
            timestamp = int(date.timestamp())
            historical_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&appid={API_KEY}&units=metric"
            response = requests.get(historical_url)
            if response.status_code == 200:
                data = response.json()
                for hour_data in data.get('hourly', []):
                    historical_data.append({
                        'datetime': datetime.datetime.fromtimestamp(hour_data['dt']),
                        'temp': hour_data['temp'],
                        'humidity': hour_data['humidity'],
                        'wind_speed': hour_data['wind_speed'],
                        'pressure': hour_data['pressure'],
                        'description': hour_data['weather'][0]['description'],
                        'main_weather': hour_data['weather'][0]['main']
                    })
        
        return pd.DataFrame(historical_data) if historical_data else None
    except Exception as e:
        st.warning(f"Could not fetch historical data: {str(e)}")
        return None

# NEW FUNCTION: Get extended 15-day forecast
def get_extended_forecast(lat, lon):
    try:
        extended_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={API_KEY}&units=metric"
        response = requests.get(extended_url)
        if response.status_code == 200:
            data = response.json()
            extended_data = []
            
            for day_data in data.get('daily', []):
                extended_data.append({
                    'date': datetime.datetime.fromtimestamp(day_data['dt']),
                    'temp_day': day_data['temp']['day'],
                    'temp_min': day_data['temp']['min'],
                    'temp_max': day_data['temp']['max'],
                    'humidity': day_data['humidity'],
                    'wind_speed': day_data['wind_speed'],
                    'description': day_data['weather'][0]['description'],
                    'main_weather': day_data['weather'][0]['main'],
                    'uv_index': day_data.get('uvi', 0),
                    'rain_prob': day_data.get('pop', 0) * 100
                })
            
            return pd.DataFrame(extended_data) if extended_data else None
    except Exception as e:
        st.warning(f"Could not fetch extended forecast: {str(e)}")
        return None

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

def get_csv_download_link(df, filename="weather_forecast.csv"):
    """Generate a link to download the DataFrame as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def prepare_lstm_data(df, features=['temp', 'humidity', 'wind_speed', 'pressure'], look_back=8):
    """Prepare data for LSTM model with multiple features."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df[features].values)
    
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        y.append(dataset[i + look_back, :])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

def create_lstm_model(look_back=8, n_features=4):
    """Create and compile LSTM model for multiple features."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, n_features)))
    model.add(LSTM(50))
    model.add(Dense(n_features))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def predict_weather_attributes(df, look_back=8, forecast_steps=40):
    """Predict multiple weather attributes for the next 5 days."""
    try:
        features = ['temp', 'humidity', 'wind_speed', 'pressure']
        X, y, scaler = prepare_lstm_data(df, features, look_back)
        
        # Train model
        model = create_lstm_model(look_back, len(features))
        model.fit(X, y, epochs=50, batch_size=1, verbose=0)
        
        # Prepare last sequence for prediction
        last_sequence = df[features].values[-look_back:]
        last_sequence = scaler.transform(last_sequence)
        last_sequence = np.reshape(last_sequence, (1, look_back, len(features)))
        
        # Predict next 40 time steps (5 days at 3-hour intervals)
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_steps):
            pred = model.predict(current_sequence, verbose=0)
            predictions.append(pred[0])
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = pred[0]
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions))
        
        # Create prediction DataFrame
        last_datetime = df['datetime'].iloc[-1]
        prediction_times = [last_datetime + datetime.timedelta(hours=3*(i+1)) for i in range(forecast_steps)]
        pred_df = pd.DataFrame({
            'datetime': prediction_times,
            'temp': predictions[:, 0],
            'humidity': predictions[:, 1],
            'wind_speed': predictions[:, 2],
            'pressure': predictions[:, 3]
        })
        
        return pred_df
    except Exception as e:
        st.warning(f"⚠️ Could not generate ML predictions: {str(e)}")
        return None

# NEW FUNCTION: Generate weather recommendations based on forecast
def generate_recommendations(df, current_weather):
    recommendations = []
    
    # Current weather recommendations
    temp = current_weather['main']['temp']
    humidity = current_weather['main']['humidity']
    weather_condition = current_weather['weather'][0]['main']
    wind_speed = current_weather['wind']['speed']
    
    # Temperature-based recommendations
    if temp < 10:
        recommendations.append("🧥 Temperature is low. Wear warm clothes and consider carrying a jacket.")
    elif temp > 30:
        recommendations.append("☀️ Temperature is high. Stay hydrated and use sunscreen if going outside.")
    
    # Weather condition recommendations
    if weather_condition == "Rain":
        recommendations.append("☔ Rain expected. Don't forget to carry an umbrella.")
    elif weather_condition == "Snow":
        recommendations.append("❄️ Snowfall expected. Drive carefully and wear appropriate footwear.")
    elif weather_condition == "Thunderstorm":
        recommendations.append("⚡ Thunderstorm expected. Avoid outdoor activities.")
    
    # Check for rain in the forecast (next 24 hours)
    next_24h = df.iloc[:8]
    if "Rain" in next_24h['main_weather'].values:
        rain_time = next_24h[next_24h['main_weather'] == "Rain"]['datetime'].iloc[0]
        time_diff = rain_time - datetime.datetime.now()
        hours = round(time_diff.total_seconds() / 3600)
        if hours > 0:
            recommendations.append(f"🌧️ Rain expected in about {hours} hours. Plan accordingly.")
    
    # UV index recommendation
    if 'uv_index' in df.columns and df['uv_index'].max() > 5:
        recommendations.append("☀️ High UV index expected. Use sunscreen and wear protective clothing.")
    
    # Wind recommendations
    if wind_speed > 10:
        recommendations.append("💨 Strong winds expected. Secure loose objects outdoors.")
    
    # Humidity recommendations
    if humidity > 80:
        recommendations.append("💧 High humidity. Stay hydrated and wear light clothing.")
    elif humidity < 30:
        recommendations.append("🏜️ Low humidity. Keep your skin moisturized.")
    
    return recommendations

# NEW FUNCTION: Translate text using Google Translate
def translate_text(text, target_language='hi'):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        return text  # Return original text if translation fails

def setup_sidebar():
    """Setup the sidebar with city selection and options."""
    st.sidebar.title("🌍 City Selection")
    
    popular_cities = {
        "New York 🗽": "New York",
        "London 🇬🇧": "London",
        "Tokyo 🗼": "Tokyo",
        "Paris 🗼": "Paris",
        "Sydney 🦘": "Sydney",
        "Singapore 🇸🇬": "Singapore",
        "Mumbai 🇮🇳": "Mumbai",
        "Delhi 🇮🇳": "Delhi",
        "Bangalore 🇮🇳": "Bangalore",
        "Kolkata 🇮🇳": "Kolkata",
        "Chennai 🇮🇳": "Chennai"
    }
    
    city_option = st.sidebar.radio("Select city input method:", ["Choose from popular cities 🌍", "Enter custom city 🔍", "Search by coordinates 🧭"])
    
    city = None
    lat = None
    lon = None
    
    if city_option == "Choose from popular cities 🌍":
        city = st.sidebar.selectbox("Select a city:", list(popular_cities.keys()))
        city = popular_cities[city]
    elif city_option == "Enter custom city 🔍":
        city = st.sidebar.text_input("Enter city name:", "")
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=28.6139, format="%.4f")
        with col2:
            lon = st.number_input("Longitude", value=77.2090, format="%.4f")
        
        # Reverse geocode to get city name
        if lat and lon:
            try:
                geocode_url = f"https://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={API_KEY}"
                response = requests.get(geocode_url)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        city = data[0]['name']
                        st.sidebar.success(f"Found location: {city}")
            except:
                pass
    
    # Background color customization
    st.sidebar.title("🎨 Customization")
    bg_color = st.sidebar.color_picker("Choose background color:", "#0029FF")
    
    # Temperature unit and language options
    temp_unit = st.sidebar.selectbox("🌡️ Temperature Unit:", ["Celsius", "Fahrenheit"])
    
    # NEW: Language selection
    language = st.sidebar.selectbox("🌐 Language:", [
        "English", "Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati"
    ])
    
    language_code = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
        "Marathi": "mr",
        "Gujarati": "gu"
    }
    
    chart_theme = st.sidebar.selectbox("🎨 Chart Theme:", [
        "Vibrant", "Pastel", "Neon", "Ocean", "Sunset"
    ])
    
    return city, lat, lon, chart_theme, bg_color, temp_unit, language_code[language]

def dashboard_page(current_weather, forecast_data, chart_theme, language_code, recommendations):
    """Main dashboard page with overview metrics."""
    if language_code != 'en':
        st.title(translate_text("🌤️ Weather Dashboard Overview", language_code))
    else:
        st.title("🌤️ Weather Dashboard Overview")
    
    weather_condition = current_weather['weather'][0]['main']
    color_scheme = get_weather_theme(weather_condition)
    
    # Weather animation
    lottie_weather = load_lottie_url(weather_condition)
    if lottie_weather:
        st_lottie(lottie_weather, height=200, key="weather_animation")
    
    # Location and time information
    city_name = current_weather['name']
    country = current_weather.get('sys', {}).get('country', '')
    current_time = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")
    
    if language_code != 'en':
        st.subheader(translate_text(f"Location: {city_name}, {country}", language_code))
        st.subheader(translate_text(f"Last Updated: {current_time}", language_code))
    else:
        st.subheader(f"Location: {city_name}, {country}")
        st.subheader(f"Last Updated: {current_time}")
    
    # Current weather metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp_label = translate_text("Temperature", language_code) if language_code != 'en' else "Temperature"
        create_colorful_metric(
            temp_label,
            f"{current_weather['main']['temp']}°C",
            f"{current_weather['main']['temp_max'] - current_weather['main']['temp_min']:+.1f}°C",
            color_scheme
        )
    
    with col2:
        humidity_label = translate_text("Humidity", language_code) if language_code != 'en' else "Humidity"
        create_colorful_metric(
            humidity_label,
            f"{current_weather['main']['humidity']}%",
            "Normal" if 30 <= current_weather['main']['humidity'] <= 70 else "Extreme",
            color_scheme
        )
    
    with col3:
        wind_label = translate_text("Wind Speed", language_code) if language_code != 'en' else "Wind Speed"
        create_colorful_metric(
            wind_label,
            f"{current_weather['wind']['speed']} m/s",
            f"Gusts: {current_weather['wind'].get('gust', 0)} m/s",
            color_scheme
        )
    
    # Additional weather metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feels_like_label = translate_text("Feels Like", language_code) if language_code != 'en' else "Feels Like"
        create_colorful_metric(
            feels_like_label,
            f"{current_weather['main']['feels_like']}°C",
            None,
            color_scheme
        )
    
    with col2:
        pressure_label = translate_text("Pressure", language_code) if language_code != 'en' else "Pressure"
        create_colorful_metric(
            pressure_label,
            f"{current_weather['main']['pressure']} hPa",
            None,
            color_scheme
        )
    
    with col3:
        visibility_label = translate_text("Visibility", language_code) if language_code != 'en' else "Visibility"
        visibility_km = current_weather.get('visibility', 10000) / 1000
        create_colorful_metric(
            visibility_label,
            f"{visibility_km:.1f} km",
            None,
            color_scheme
        )
    
    # Process forecast data
    df = process_forecast_data(forecast_data)
    
    # Personalized Recommendations
    st.markdown("---")
    if language_code != 'en':
        st.subheader(translate_text("🧠 Smart Recommendations", language_code))
    else:
        st.subheader("🧠 Smart Recommendations")
    
    for rec in recommendations:
        if language_code != 'en':
            st.info(translate_text(rec, language_code))
        else:
            st.info(rec)
    
    # Mini charts in columns
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        temp_trend_label = translate_text("🌡️ Current Temperature Trend", language_code) if language_code != 'en' else "🌡️ Current Temperature Trend"
        st.subheader(temp_trend_label)
        fig_temp_mini = px.line(df.head(8), x='datetime', y='temp', 
                               color_discrete_sequence=px.colors.sequential.Sunset)
        st.plotly_chart(fig_temp_mini, use_container_width=True)
    
    with col2:
        humidity_trend_label = translate_text("💧 Humidity Trend", language_code) if language_code != 'en' else "💧 Humidity Trend"
        st.subheader(humidity_trend_label)
        fig_humidity_mini = px.area(df.head(8), x='datetime', y='humidity', 
                                   color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig_humidity_mini, use_container_width=True)
    
    # NEW: Quick weather summary
    st.markdown("---")
    summary_title = translate_text("Weather Summary", language_code) if language_code != 'en' else "Weather Summary"
    st.subheader(summary_title)
    
    weather_desc = current_weather['weather'][0]['description'].capitalize()
    if language_code != 'en':
        weather_desc = translate_text(weather_desc, language_code)
    
    summary_text = f"""
    **{weather_desc}** with a temperature of **{current_weather['main']['temp']}°C**.
    The day's high will be **{current_weather['main']['temp_max']}°C** and the low will be **{current_weather['main']['temp_min']}°C**.
    The humidity is at **{current_weather['main']['humidity']}%** with a wind speed of **{current_weather['wind']['speed']} m/s**.
    """
    
    if language_code != 'en':
        st.markdown(translate_text(summary_text, language_code))
    else:
        st.markdown(summary_text)
    
    # Navigation Menu
    st.markdown("---")
    nav_title = translate_text("Quick Navigation", language_code) if language_code != 'en' else "Quick Navigation"
    st.subheader(nav_title)
    
    nav_links = [
        "📊 Weather Analytics",
        "🌡️ Temperature Forecast",
        "💧 Humidity & Wind",
        "🌍 Maps & Visualizations",
        "🤖 ML Predictions",
        "📚 Historical Data",
        "📈 Extended Forecast",
        "📱 Weather Alerts",
        "📥 Data Export"
    ]
    
    if language_code != 'en':
        translated_links = [translate_text(link, language_code) for link in nav_links]
        st.markdown("\n".join([f"- **{link}**" for link in translated_links]))
    else:
        st.markdown("\n".join([f"- **{link}**" for link in nav_links]))

def weather_analytics_page(df, chart_theme, language_code):
    """Page for weather condition analytics."""
    if language_code != 'en':
        st.title(translate_text("📊 Weather Condition Analytics", language_code))
    else:
        st.title("📊 Weather Condition Analytics")
    
    chart_colors = {
        "Vibrant": px.colors.qualitative.Set1,
        "Pastel": px.colors.qualitative.Pastel,
        "Neon": px.colors.qualitative.Bold,
        "Ocean": px.colors.sequential.Blues,
        "Sunset": px.colors.sequential.Sunset
    }
    
    # Weather Condition Summary
    condition_title = translate_text("🌤️ Weather Condition Distribution", language_code) if language_code != 'en' else "🌤️ Weather Condition Distribution"
    st.subheader(condition_title)
    
    weather_summary = df['main_weather'].value_counts().reset_index()
    weather_summary.columns = ['Condition', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_weather_pie = px.pie(weather_summary, values='Count', names='Condition', 
                                title='Weather Condition Distribution', 
                                color_discrete_sequence=chart_colors[chart_theme])
        st.plotly_chart(fig_weather_pie, use_container_width=True)
    
    with col2:
        fig_weather_bar = px.bar(weather_summary, x='Condition', y='Count', 
                                title='Weather Condition Counts',
                                color='Condition',
                                color_discrete_sequence=chart_colors[chart_theme])
        st.plotly_chart(fig_weather_bar, use_container_width=True)
    
    # Weather Over Time
    timeline_title = translate_text("⏳ Weather Conditions Over Time", language_code) if language_code != 'en' else "⏳ Weather Conditions Over Time"
    st.subheader(timeline_title)
    
    fig_weather_timeline = px.scatter(df, x='datetime', y='temp', color='main_weather',
                                     title='Temperature with Weather Conditions',
                                     color_discrete_sequence=chart_colors[chart_theme])
    st.plotly_chart(fig_weather_timeline, use_container_width=True)
    
    # NEW: Weather Description Word Cloud
    word_cloud_title = translate_text("🔤 Weather Description Analysis", language_code) if language_code != 'en' else "🔤 Weather Description Analysis"
    st.subheader(word_cloud_title)
    
    # Create a simple table showing descriptions and their frequencies
    descriptions = df['description'].value_counts().reset_index()
    descriptions.columns = ['Description', 'Frequency']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(descriptions)
    
    with col2:
        # Create a horizontal bar chart for descriptions
        fig_desc = px.bar(descriptions.head(10), x='Frequency', y='Description', 
                         orientation='h', color='Frequency',
                                                 color_continuous_scale=chart_colors[chart_theme],
                        title='Most Common Weather Descriptions')
        st.plotly_chart(fig_desc, use_container_width=True)

    # Weather Condition Correlations
    correlation_title = translate_text("📈 Weather Parameter Correlations", language_code) if language_code != 'en' else "📈 Weather Parameter Correlations"
    st.subheader(correlation_title)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=chart_colors[chart_theme],
                        title='Correlation Between Weather Parameters')
    st.plotly_chart(fig_corr, use_container_width=True)

def temperature_forecast_page(df, chart_theme, language_code):
    """Page for temperature forecast visualization."""
    if language_code != 'en':
        st.title(translate_text("🌡️ Temperature Forecast", language_code))
    else:
        st.title("🌡️ Temperature Forecast")
    
    # Temperature Trends
    trend_title = translate_text("📈 Temperature Trend (Next 5 Days)", language_code) if language_code != 'en' else "📈 Temperature Trend (Next 5 Days)"
    st.subheader(trend_title)
    
    fig_temp = px.line(df, x='datetime', y=['temp', 'temp_min', 'temp_max'],
                      title='Temperature Forecast',
                      labels={'value': 'Temperature (°C)', 'variable': 'Metric'},
                      color_discrete_sequence=px.colors.sequential.Sunset)
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Temperature Distribution
    dist_title = translate_text("📊 Temperature Distribution", language_code) if language_code != 'en' else "📊 Temperature Distribution"
    st.subheader(dist_title)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temp_dist = px.histogram(df, x='temp', nbins=20,
                                    title='Temperature Distribution',
                                    color_discrete_sequence=[px.colors.sequential.Sunset[2]])
        st.plotly_chart(fig_temp_dist, use_container_width=True)
    
    with col2:
        fig_box = px.box(df, y='temp',
                        title='Temperature Range',
                        color_discrete_sequence=[px.colors.sequential.Sunset[4]])
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Daily Temperature Extremes
    extremes_title = translate_text("🔥❄️ Daily Temperature Extremes", language_code) if language_code != 'en' else "🔥❄️ Daily Temperature Extremes"
    st.subheader(extremes_title)
    
    df['date'] = df['datetime'].dt.date
    daily_extremes = df.groupby('date').agg({'temp_max': 'max', 'temp_min': 'min'}).reset_index()
    
    fig_extremes = go.Figure()
    fig_extremes.add_trace(go.Scatter(
        x=daily_extremes['date'], y=daily_extremes['temp_max'],
        name='Daily High',
        line=dict(color='red', width=2)
    ))
    fig_extremes.add_trace(go.Scatter(
        x=daily_extremes['date'], y=daily_extremes['temp_min'],
        name='Daily Low',
        line=dict(color='blue', width=2)
    ))
    fig_extremes.update_layout(
        title='Daily High and Low Temperatures',
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        template='plotly_white'
    )
    st.plotly_chart(fig_extremes, use_container_width=True)

def humidity_wind_page(df, chart_theme, language_code):
    """Page for humidity and wind analysis."""
    if language_code != 'en':
        st.title(translate_text("💨 Humidity & Wind Analysis", language_code))
    else:
        st.title("💨 Humidity & Wind Analysis")
    
    # Humidity Trends
    humidity_title = translate_text("💧 Humidity Trend", language_code) if language_code != 'en' else "💧 Humidity Trend"
    st.subheader(humidity_title)
    
    fig_humidity = px.line(df, x='datetime', y='humidity',
                          title='Humidity Forecast',
                          labels={'value': 'Humidity (%)'},
                          color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(fig_humidity, use_container_width=True)
    
    # Wind Analysis
    wind_title = translate_text("🌬️ Wind Speed and Direction", language_code) if language_code != 'en' else "🌬️ Wind Speed and Direction"
    st.subheader(wind_title)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_wind_speed = px.line(df, x='datetime', y='wind_speed',
                                title='Wind Speed Forecast',
                                labels={'value': 'Wind Speed (m/s)'},
                                color_discrete_sequence=px.colors.sequential.Greens)
        st.plotly_chart(fig_wind_speed, use_container_width=True)
    
    with col2:
        fig_wind_rose = px.bar_polar(df, r='wind_speed', theta='wind_deg',
                                    title='Wind Direction Distribution',
                                    color='wind_speed',
                                    color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_wind_rose, use_container_width=True)
    
    # Wind vs Humidity
    relation_title = translate_text("💨💧 Wind Speed vs Humidity", language_code) if language_code != 'en' else "💨💧 Wind Speed vs Humidity"
    st.subheader(relation_title)
    
    fig_wind_humidity = px.scatter(df, x='wind_speed', y='humidity',
                                  color='temp',
                                  size='pressure',
                                  title='Wind Speed vs Humidity (Colored by Temperature)',
                                  color_continuous_scale=px.colors.sequential.Sunset)
    st.plotly_chart(fig_wind_humidity, use_container_width=True)

def maps_visualization_page(current_weather, forecast_df, language_code):
    """Page for map visualizations."""
    if language_code != 'en':
        st.title(translate_text("🌍 Maps & Visualizations", language_code))
    else:
        st.title("🌍 Maps & Visualizations")
    
    lat = current_weather['coord']['lat']
    lon = current_weather['coord']['lon']
    temp = current_weather['main']['temp']
    city_name = current_weather['name']
    
    # Heatmap
    heatmap_title = translate_text("🔥 Temperature Heatmap", language_code) if language_code != 'en' else "🔥 Temperature Heatmap"
    st.subheader(heatmap_title)
    
    heat_map = create_heatmap(lat, lon, temp)
    folium_static(heat_map)
    
    # City Area Map
    area_title = translate_text("🏙️ City Area Overview", language_code) if language_code != 'en' else "🏙️ City Area Overview"
    st.subheader(area_title)
    
    city_map = create_city_area_map(lat, lon, city_name)
    folium_static(city_map)
    
    # Weather Stations Map (simulated)
    stations_title = translate_text("📡 Nearby Weather Stations", language_code) if language_code != 'en' else "📡 Nearby Weather Stations"
    st.subheader(stations_title)
    
    # Simulate nearby stations
    stations_map = folium.Map(location=[lat, lon], zoom_start=10)
    
    # Add main station
    folium.Marker(
        location=[lat, lon],
        popup=f"Main Station: {temp}°C",
        icon=folium.Icon(color='red')
    ).add_to(stations_map)
    
    # Add simulated nearby stations
    for i in range(5):
        station_lat = lat + np.random.uniform(-0.2, 0.2)
        station_lon = lon + np.random.uniform(-0.2, 0.2)
        station_temp = temp + np.random.uniform(-3, 3)
        
        folium.Marker(
            location=[station_lat, station_lon],
            popup=f"Station {i+1}: {station_temp:.1f}°C",
            icon=folium.Icon(color='blue')
        ).add_to(stations_map)
    
    folium_static(stations_map)

def ml_predictions_page(df, language_code):
    """Page for machine learning predictions."""
    if language_code != 'en':
        st.title(translate_text("🤖 Machine Learning Predictions", language_code))
    else:
        st.title("🤖 Machine Learning Predictions")
    
    prediction_title = translate_text("🔮 5-Day Weather Forecast Prediction", language_code) if language_code != 'en' else "🔮 5-Day Weather Forecast Prediction"
    st.subheader(prediction_title)
    
    with st.spinner(translate_text("Training model and generating predictions...", language_code) if language_code != 'en' else "Training model and generating predictions..."):
        pred_df = predict_weather_attributes(df)
    
    if pred_df is not None:
        st.success(translate_text("Predictions generated successfully!", language_code) if language_code != 'en' else "Predictions generated successfully!")
        
        # Show predictions
        st.dataframe(pred_df)
        
        # Plot predictions
        fig_pred_temp = px.line(pred_df, x='datetime', y=['temp', 'humidity', 'wind_speed', 'pressure'],
                               title='Predicted Weather Parameters',
                               labels={'value': 'Value', 'variable': 'Parameter'},
                               color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pred_temp, use_container_width=True)
        
        # Model explanation
        explanation_title = translate_text("🧠 How the Predictions Work", language_code) if language_code != 'en' else "🧠 How the Predictions Work"
        st.subheader(explanation_title)
        
        explanation_text = """
        The predictions are generated using an LSTM (Long Short-Term Memory) neural network, which is particularly 
        effective for time series forecasting. The model was trained on:
        - Past temperature data
        - Humidity levels
        - Wind speed
        - Atmospheric pressure
        
        The model looks at patterns in the last 24 hours (8 data points at 3-hour intervals) to predict 
        the next 5 days (40 data points at 3-hour intervals).
        """
        
        if language_code != 'en':
            st.markdown(translate_text(explanation_text, language_code))
        else:
            st.markdown(explanation_text)
    else:
        st.warning(translate_text("Could not generate predictions. Please try again later.", language_code) if language_code != 'en' else "Could not generate predictions. Please try again later.")

def historical_data_page(current_weather, language_code):
    """Page for historical weather data."""
    if language_code != 'en':
        st.title(translate_text("📚 Historical Weather Data", language_code))
    else:
        st.title("📚 Historical Weather Data")
    
    lat = current_weather['coord']['lat']
    lon = current_weather['coord']['lon']
    
    with st.spinner(translate_text("Fetching historical weather data...", language_code) if language_code != 'en' else "Fetching historical weather data..."):
        historical_df = get_historical_weather(lat, lon)
    
    if historical_df is not None and not historical_df.empty:
        st.success(translate_text("Historical data loaded successfully!", language_code) if language_code != 'en' else "Historical data loaded successfully!")
        
        # Show last 7 days data
        st.subheader(translate_text("⏳ Last 7 Days Weather", language_code) if language_code != 'en' else "⏳ Last 7 Days Weather")
        st.dataframe(historical_df)
        
        # Plot historical trends
        fig_hist_temp = px.line(historical_df, x='datetime', y='temp',
                               title='Temperature Trend (Last 7 Days)',
                               labels={'value': 'Temperature (°C)'})
        st.plotly_chart(fig_hist_temp, use_container_width=True)
        
        fig_hist_humidity = px.line(historical_df, x='datetime', y='humidity',
                                   title='Humidity Trend (Last 7 Days)',
                                   labels={'value': 'Humidity (%)'})
        st.plotly_chart(fig_hist_humidity, use_container_width=True)
        
        # Weather condition analysis
        st.subheader(translate_text("🌤️ Historical Weather Conditions", language_code) if language_code != 'en' else "🌤️ Historical Weather Conditions")
        
        weather_counts = historical_df['main_weather'].value_counts().reset_index()
        weather_counts.columns = ['Condition', 'Count']
        
        fig_hist_weather = px.bar(weather_counts, x='Condition', y='Count',
                                 title='Weather Condition Frequency',
                                 color='Condition')
        st.plotly_chart(fig_hist_weather, use_container_width=True)
    else:
        st.warning(translate_text("Could not fetch historical data. This feature might not be available for all locations.", language_code) if language_code != 'en' else "Could not fetch historical data. This feature might not be available for all locations.")

def extended_forecast_page(current_weather, language_code):
    """Page for extended 15-day forecast."""
    if language_code != 'en':
        st.title(translate_text("📈 Extended 15-Day Forecast", language_code))
    else:
        st.title("📈 Extended 15-Day Forecast")
    
    lat = current_weather['coord']['lat']
    lon = current_weather['coord']['lon']
    
    with st.spinner(translate_text("Fetching extended forecast...", language_code) if language_code != 'en' else "Fetching extended forecast..."):
        extended_df = get_extended_forecast(lat, lon)
    
    if extended_df is not None and not extended_df.empty:
        st.success(translate_text("Extended forecast loaded successfully!", language_code) if language_code != 'en' else "Extended forecast loaded successfully!")
        
        # Show extended forecast
        st.dataframe(extended_df)
        
        # Plot extended forecast
        fig_ext_temp = px.line(extended_df, x='date', y=['temp_day', 'temp_min', 'temp_max'],
                             title='15-Day Temperature Forecast',
                             labels={'value': 'Temperature (°C)', 'variable': 'Metric'},
                             color_discrete_sequence=px.colors.sequential.Sunset)
        st.plotly_chart(fig_ext_temp, use_container_width=True)
        
        # UV Index and Rain Probability
        st.subheader(translate_text("☀️ UV Index & 🌧️ Rain Probability", language_code) if language_code != 'en' else "☀️ UV Index & 🌧️ Rain Probability")
        
        fig_uv_rain = go.Figure()
        fig_uv_rain.add_trace(go.Bar(
            x=extended_df['date'],
            y=extended_df['uv_index'],
            name='UV Index',
            marker_color='orange'
        ))
        fig_uv_rain.add_trace(go.Scatter(
            x=extended_df['date'],
            y=extended_df['rain_prob'],
            name='Rain Probability (%)',
            yaxis='y2',
            line=dict(color='blue')
        ))
        fig_uv_rain.update_layout(
            yaxis=dict(title='UV Index'),
            yaxis2=dict(title='Rain Probability (%)', overlaying='y', side='right'),
            title='UV Index and Rain Probability Forecast'
        )
        st.plotly_chart(fig_uv_rain, use_container_width=True)
    else:
        st.warning(translate_text("Could not fetch extended forecast. Please try again later.", language_code) if language_code != 'en' else "Could not fetch extended forecast. Please try again later.")

def weather_alerts_page(current_weather, forecast_df, language_code):
    """Page for weather alerts and warnings."""
    if language_code != 'en':
        st.title(translate_text("⚠️ Weather Alerts & Warnings", language_code))
    else:
        st.title("⚠️ Weather Alerts & Warnings")
    
    # Check for extreme weather conditions
    alerts = []
    
    # Temperature alerts
    if current_weather['main']['temp'] > 35:
        alerts.append(translate_text("🌡️ High Temperature Warning: Extreme heat expected. Stay hydrated and avoid prolonged sun exposure.", language_code))
    elif current_weather['main']['temp'] < 0:
        alerts.append(translate_text("❄️ Low Temperature Warning: Freezing temperatures expected. Dress warmly and protect exposed skin.", language_code))
    
    # Wind alerts
    if current_weather['wind']['speed'] > 10:
        alerts.append(translate_text("💨 High Wind Warning: Strong winds expected. Secure loose objects and be cautious outdoors.", language_code))
    
    # Precipitation alerts
    if 'Rain' in forecast_df['main_weather'].values:
        alerts.append(translate_text("🌧️ Rain Alert: Precipitation expected. Carry an umbrella and allow extra travel time.", language_code))
    if 'Snow' in forecast_df['main_weather'].values:
        alerts.append(translate_text("❄️ Snow Alert: Snowfall expected. Drive carefully and allow extra travel time.", language_code))
    if 'Thunderstorm' in forecast_df['main_weather'].values:
        alerts.append(translate_text("⚡ Thunderstorm Alert: Potential thunderstorms expected. Seek indoor shelter if storms approach.", language_code))
    
    # UV index alerts
    if 'uv_index' in forecast_df.columns and forecast_df['uv_index'].max() > 8:
        alerts.append(translate_text("☀️ Extreme UV Alert: Very high UV index expected. Use sunscreen and wear protective clothing.", language_code))
    
    # Display alerts or "all clear" message
    if alerts:
        st.warning(translate_text("⚠️ Active Weather Alerts:", language_code) if language_code != 'en' else "⚠️ Active Weather Alerts:")
        for alert in alerts:
            st.error(alert)
    else:
        st.success(translate_text("✅ No severe weather alerts for your area at this time.", language_code) if language_code != 'en' else "✅ No severe weather alerts for your area at this time.")
    
    # Weather advisory information
    advisory_title = translate_text("ℹ️ Weather Advisory Information", language_code) if language_code != 'en' else "ℹ️ Weather Advisory Information"
    st.subheader(advisory_title)
    
    advisory_text = translate_text("""
    Weather alerts are generated based on:
    - Extreme temperature conditions
    - High wind speeds
    - Precipitation events
    - High UV index levels
    
    Always check local weather sources for the most up-to-date information, especially during severe weather events.
    """, language_code) if language_code != 'en' else """
    Weather alerts are generated based on:
    - Extreme temperature conditions
    - High wind speeds
    - Precipitation events
    - High UV index levels
    
    Always check local weather sources for the most up-to-date information, especially during severe weather events.
    """
    
    st.info(advisory_text)

def data_export_page(current_weather, forecast_df, historical_df, extended_df, language_code):
    """Page for data export options."""
    if language_code != 'en':
        st.title(translate_text("📥 Data Export Options", language_code))
    else:
        st.title("📥 Data Export Options")
    
    # Current weather data
    st.subheader(translate_text("🌤️ Current Weather Data", language_code) if language_code != 'en' else "🌤️ Current Weather Data")
    current_df = pd.DataFrame([current_weather])
    st.dataframe(current_df)
    
    st.download_button(
        label=translate_text("Download Current Weather Data (CSV)", language_code) if language_code != 'en' else "Download Current Weather Data (CSV)",
        data=current_df.to_csv(index=False).encode('utf-8'),
        file_name='current_weather.csv',
        mime='text/csv'
    )
    
    # 5-day forecast data
    st.subheader(translate_text("📅 5-Day Forecast Data", language_code) if language_code != 'en' else "📅 5-Day Forecast Data")
    st.dataframe(forecast_df)
    
    st.download_button(
        label=translate_text("Download Forecast Data (CSV)", language_code) if language_code != 'en' else "Download Forecast Data (CSV)",
        data=forecast_df.to_csv(index=False).encode('utf-8'),
        file_name='weather_forecast.csv',
        mime='text/csv'
    )
    
    # Historical data if available
    if historical_df is not None and not historical_df.empty:
        st.subheader(translate_text("⏳ Historical Weather Data", language_code) if language_code != 'en' else "⏳ Historical Weather Data")
        st.dataframe(historical_df)
        
        st.download_button(
            label=translate_text("Download Historical Data (CSV)", language_code) if language_code != 'en' else "Download Historical Data (CSV)",
            data=historical_df.to_csv(index=False).encode('utf-8'),
            file_name='historical_weather.csv',
            mime='text/csv'
        )
    
    # Extended forecast if available
    if extended_df is not None and not extended_df.empty:
        st.subheader(translate_text("📈 Extended Forecast Data", language_code) if language_code != 'en' else "📈 Extended Forecast Data")
        st.dataframe(extended_df)
        
        st.download_button(
            label=translate_text("Download Extended Forecast (CSV)", language_code) if language_code != 'en' else "Download Extended Forecast (CSV)",
            data=extended_df.to_csv(index=False).encode('utf-8'),
            file_name='extended_forecast.csv',
            mime='text/csv'
        )
    
    # JSON export option
    st.subheader(translate_text("📤 Export All Data as JSON", language_code) if language_code != 'en' else "📤 Export All Data as JSON")
    
    all_data = {
        'current_weather': current_weather,
        'forecast': forecast_df.to_dict('records'),
        'historical': historical_df.to_dict('records') if historical_df is not None else None,
        'extended_forecast': extended_df.to_dict('records') if extended_df is not None else None
    }
    
    st.download_button(
        label=translate_text("Download All Data (JSON)", language_code) if language_code != 'en' else "Download All Data (JSON)",
        data=json.dumps(all_data, indent=2),
        file_name='weather_data.json',
        mime='application/json'
    )

def main():
    """Main function to run the Streamlit app."""
    # Setup sidebar and get user inputs
    city, lat, lon, chart_theme, bg_color, temp_unit, language_code = setup_sidebar()
    
    # Apply custom background color
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            background-image: linear-gradient(to bottom, {bg_color}, #ffffff);
            opacity: 0.9;
        }}
        .metric-card {{
            background: linear-gradient(45deg, #FF69B4, #DA70D6);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
        }}
        .metric-card:hover {{
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Check if we have city or coordinates
    if (city is None or city == "") and (lat is None or lon is None):
        st.warning(translate_text("Please enter a city name or coordinates to get weather information.", language_code) if language_code != 'en' else "Please enter a city name or coordinates to get weather information.")
        return
    
    # Get weather data
    if city:
        current_weather, forecast_data = get_weather_data(city)
    else:
        try:
            current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            current_response = requests.get(current_url)
            forecast_response = requests.get(forecast_url)
            current_weather = current_response.json()
            forecast_data = forecast_response.json()
        except:
            st.error(translate_text("Error fetching weather data for the provided coordinates.", language_code) if language_code != 'en' else "Error fetching weather data for the provided coordinates.")
            return
    
    if current_weather is None or forecast_data is None:
        return
    
    # Process forecast data
    forecast_df = process_forecast_data(forecast_data)
    
    # Get historical and extended forecast data
    historical_df = get_historical_weather(current_weather['coord']['lat'], current_weather['coord']['lon'])
    extended_df = get_extended_forecast(current_weather['coord']['lat'], current_weather['coord']['lon'])
    
    # Generate recommendations
    recommendations = generate_recommendations(forecast_df, current_weather)
    
    # Navigation menu
    st.sidebar.title("📱 Navigation")
    page_options = {
        translate_text("🌤️ Dashboard Overview", language_code): "dashboard",
        translate_text("📊 Weather Analytics", language_code): "analytics",
        translate_text("🌡️ Temperature Forecast", language_code): "temperature",
        translate_text("💨 Humidity & Wind", language_code): "humidity_wind",
        translate_text("🌍 Maps & Visualizations", language_code): "maps",
        translate_text("🤖 ML Predictions", language_code): "ml_predictions",
        translate_text("📚 Historical Data", language_code): "historical",
        translate_text("📈 Extended Forecast", language_code): "extended",
        translate_text("⚠️ Weather Alerts", language_code): "alerts",
        translate_text("📥 Data Export", language_code): "export"
    }
    
    selected_page = st.sidebar.radio(
        translate_text("Go to:", language_code),
        list(page_options.keys())
    )
    
    # Display the selected page
    if page_options[selected_page] == "dashboard":
        dashboard_page(current_weather, forecast_data, chart_theme, language_code, recommendations)
    elif page_options[selected_page] == "analytics":
        weather_analytics_page(forecast_df, chart_theme, language_code)
    elif page_options[selected_page] == "temperature":
        temperature_forecast_page(forecast_df, chart_theme, language_code)
    elif page_options[selected_page] == "humidity_wind":
        humidity_wind_page(forecast_df, chart_theme, language_code)
    elif page_options[selected_page] == "maps":
        maps_visualization_page(current_weather, forecast_df, language_code)
    elif page_options[selected_page] == "ml_predictions":
        ml_predictions_page(forecast_df, language_code)
    elif page_options[selected_page] == "historical":
        historical_data_page(current_weather, language_code)
    elif page_options[selected_page] == "extended":
        extended_forecast_page(current_weather, language_code)
    elif page_options[selected_page] == "alerts":
        weather_alerts_page(current_weather, forecast_df, language_code)
    elif page_options[selected_page] == "export":
        data_export_page(current_weather, forecast_df, historical_df, extended_df, language_code)

if __name__ == "__main__":
    main()                         