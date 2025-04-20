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
    
    # Approximate city area with a simple polygon (square around center)
    # This is a simplified approach; real city boundaries would require GeoJSON data
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
            
            # 3D Pressure Trends
            st.subheader("🌍 3D Pressure Trends")
            fig_pressure_3d = go.Figure(
                data=[go.Scatter3d(
                    x=df['datetime'], 
                    y=df['pressure'], 
                    z=df['wind_speed'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=df['pressure'],
                        colorscale='viridis',
                        opacity=1
                    )
                )]
            )
            fig_pressure_3d.update_layout(
                title="3D Pressure Trends",
                scene=dict(
                    xaxis_title="Datetime",
                    yaxis_title="Pressure (hPa)",
                    zaxis_title="Wind Speed (m/s)"
                )
            )
            st.plotly_chart(fig_pressure_3d, use_container_width=True)
            
            # ML Predictions for all attributes
            st.subheader("🤖 ML Weather Predictions (Next 5 Days)")
            pred_df = predict_weather_attributes(df)
            if pred_df is not None:
                # Plot predictions for each attribute
                for attr, title, y_label in [
                    ('temp', 'Temperature Prediction', 'Temperature (°C)'),
                    ('humidity', 'Humidity Prediction', 'Humidity (%)'),
                    ('wind_speed', 'Wind Speed Prediction', 'Wind Speed (m/s)'),
                    ('pressure', 'Pressure Prediction', 'Pressure (hPa)')
                ]:
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=df['datetime'],
                        y=df[attr],
                        mode='lines',
                        name='Historical',
                        line=dict(color=chart_colors[chart_theme][0])
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=pred_df['datetime'],
                        y=pred_df[attr],
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color=chart_colors[chart_theme][1], dash='dash')
                    ))
                    fig_pred.update_layout(
                        title=f'LSTM {title}',
                        xaxis_title='DateTime',
                        yaxis_title=y_label,
                        showlegend=True
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                # Download ML Predictions CSV
                st.markdown(get_csv_download_link(pred_df, "ml_weather_predictions.csv"), unsafe_allow_html=True)
            
            # Download Historical Weather Data CSV
            st.subheader("📥 Download Historical Weather Data")
            st.markdown(get_csv_download_link(df, "historical_weather_forecast.csv"), unsafe_allow_html=True)
            
            # Heatmap for the entered city
            st.subheader("🔥 Temperature Heatmap for the Entered City")
            lat = current_weather['coord']['lat']
            lon = current_weather['coord']['lon']
            temp = current_weather['main']['temp']
            heatmap = create_heatmap(lat, lon, temp)
            folium_static(heatmap, width=1200, height=600)
    

if __name__ == "__main__":
    main()