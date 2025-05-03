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

def setup_sidebar():
    """Setup the sidebar with city selection and options."""
    st.sidebar.title("🌍 City Selection")
    
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
    
    # Background color customization
    st.sidebar.title("🎨 Customization")
    bg_color = st.sidebar.color_picker("Choose background color:", "#f0f2f6")
    
    temp_unit = st.sidebar.selectbox("🌡️ Temperature Unit:", ["Celsius"])
    chart_theme = st.sidebar.selectbox("🎨 Chart Theme:", [
        "Vibrant", "Pastel", "Neon", "Ocean", "Sunset"
    ])
    
    return city, chart_theme, bg_color

def dashboard_page(current_weather, forecast_data, chart_theme):
    """Main dashboard page with overview metrics."""
    st.title("🌤️ Weather Dashboard Overview")
    
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
    
    # Mini charts in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Current Temperature Trend")
        fig_temp_mini = px.line(df.head(8), x='datetime', y='temp', 
                               color_discrete_sequence=px.colors.sequential.Sunset)
        st.plotly_chart(fig_temp_mini, use_container_width=True)
    
    with col2:
        st.subheader("💧 Humidity Trend")
        fig_humidity_mini = px.area(df.head(8), x='datetime', y='humidity', 
                                   color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig_humidity_mini, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Quick Navigation")
    st.markdown("""
    - **📊 Weather Analytics**: Detailed weather condition analysis
    - **🌡️ Temperature Forecast**: Hourly and daily temperature predictions
    - **💧 Humidity & Wind**: Humidity and wind speed analysis
    - **🌍 Maps & Visualizations**: Geographical visualizations
    - **🤖 ML Predictions**: Machine learning weather forecasts
    - **📥 Data Export**: Download weather data
    """)

def weather_analytics_page(df, chart_theme):
    """Page for weather condition analytics."""
    st.title("📊 Weather Condition Analytics")
    
    chart_colors = {
        "Vibrant": px.colors.qualitative.Set1,
        "Pastel": px.colors.qualitative.Pastel,
        "Neon": px.colors.qualitative.Bold,
        "Ocean": px.colors.sequential.Blues,
        "Sunset": px.colors.sequential.Sunset
    }
    
    # Weather Condition Summary
    st.subheader("🌤️ Weather Condition Distribution")
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
    st.subheader("⏳ Weather Conditions Over Time")
    fig_weather_timeline = px.scatter(df, x='datetime', y='temp', color='main_weather',
                                     title='Temperature with Weather Conditions',
                                     color_discrete_sequence=chart_colors[chart_theme])
    st.plotly_chart(fig_weather_timeline, use_container_width=True)

def temperature_page(df, chart_theme):
    """Page dedicated to temperature analysis."""
    st.title("🌡️ Temperature Forecast Analysis")
    
    chart_colors = {
        "Vibrant": px.colors.qualitative.Set1,
        "Pastel": px.colors.qualitative.Pastel,
        "Neon": px.colors.qualitative.Bold,
        "Ocean": px.colors.sequential.Blues,
        "Sunset": px.colors.sequential.Sunset
    }
    
    # Hourly Forecast for the next 24 hours
    st.subheader("⏳ Hourly Forecast (Next 24 Hours)")
    hourly_df = df.head(8)  # Next 24 hours (3-hour intervals)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hourly_line = px.line(hourly_df, x='datetime', y='temp', 
                                title='Hourly Temperature Forecast', 
                                color_discrete_sequence=chart_colors[chart_theme])
        st.plotly_chart(fig_hourly_line, use_container_width=True)
    
    with col2:
        fig_hourly_bar = px.bar(hourly_df, x='datetime', y='temp', 
                              title='Hourly Temperature Bars',
                              color='temp',
                              color_continuous_scale=chart_colors[chart_theme])
        st.plotly_chart(fig_hourly_bar, use_container_width=True)
    
    # Full Temperature Forecast
    st.subheader("📈 Full Temperature Forecast")
    fig_temp = px.line(df, x='datetime', y=['temp', 'temp_min', 'temp_max'], 
                      title='Temperature Forecast with Min/Max',
                      color_discrete_sequence=chart_colors[chart_theme])
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Temperature Range Area Chart
    st.subheader("🌡️ Temperature Range")
    fig_temp_range = go.Figure()
    fig_temp_range.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['temp_max'],
        fill=None,
        mode='lines',
        line_color='red',
        name='Max Temp'
    ))
    fig_temp_range.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['temp_min'],
        fill='tonexty',
        mode='lines',
        line_color='blue',
        name='Min Temp'
    ))
    fig_temp_range.update_layout(
        title='Temperature Range Forecast',
        xaxis_title='Date',
        yaxis_title='Temperature (°C)'
    )
    st.plotly_chart(fig_temp_range, use_container_width=True)

def humidity_wind_page(df, chart_theme):
    """Page for humidity and wind analysis."""
    st.title("💧 Humidity & Wind Analysis")
    
    chart_colors = {
        "Vibrant": px.colors.qualitative.Set1,
        "Pastel": px.colors.qualitative.Pastel,
        "Neon": px.colors.qualitative.Bold,
        "Ocean": px.colors.sequential.Blues,
        "Sunset": px.colors.sequential.Sunset
    }
    
    # Humidity Analysis
    st.subheader("💧 Humidity Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_humidity = px.area(df, x='datetime', y='humidity', 
                             title='Humidity Forecast', 
                             color_discrete_sequence=chart_colors[chart_theme])
        st.plotly_chart(fig_humidity, use_container_width=True)
    
    with col2:
        fig_humidity_box = px.box(df, y='humidity', 
                                title='Humidity Distribution',
                                color_discrete_sequence=chart_colors[chart_theme])
        st.plotly_chart(fig_humidity_box, use_container_width=True)
    
    # Wind Analysis
    st.subheader("🌬️ Wind Speed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_wind = px.bar(df, x='datetime', y='wind_speed', 
                         title='Wind Speed Distribution', 
                         color='wind_speed',
                         color_continuous_scale=chart_colors[chart_theme])
        st.plotly_chart(fig_wind, use_container_width=True)
    
    with col2:
        fig_wind_polar = px.line_polar(df, r='wind_speed', theta='wind_deg',
                                     title='Wind Direction and Speed',
                                     color_discrete_sequence=chart_colors[chart_theme])
        st.plotly_chart(fig_wind_polar, use_container_width=True)
    
    # Combined Humidity and Wind
    st.subheader("💨 Humidity & Wind Correlation")
    fig_combined = px.scatter(df, x='humidity', y='wind_speed', 
                            color='temp', size='pressure',
                            title='Humidity vs Wind Speed with Temperature and Pressure',
                            color_continuous_scale=chart_colors[chart_theme])
    st.plotly_chart(fig_combined, use_container_width=True)

def maps_page(current_weather, df):
    """Page for geographical visualizations."""
    st.title("🌍 Maps & Geographical Visualizations")
    
    lat = current_weather['coord']['lat']
    lon = current_weather['coord']['lon']
    temp = current_weather['main']['temp']
    city_name = current_weather['name']
    
    # Heatmap
    st.subheader("🔥 Temperature Heatmap")
    heatmap = create_heatmap(lat, lon, temp)
    folium_static(heatmap, width=1200, height=400)
    
    # City Area Map
    st.subheader("🏙️ City Area Visualization")
    city_map = create_city_area_map(lat, lon, city_name)
    folium_static(city_map, width=1200, height=400)
    
    # 3D Scatter Map
    st.subheader("🌐 3D Geographical Visualization")
    fig_3d_map = px.scatter_3d(
        df.head(24),  # First 24 data points (3 days)
        x=[lon]*24,
        y=[lat]*24,
        z='temp',
        color='temp',
        size='humidity',
        hover_name='datetime',
        title='3D Temperature Visualization Over Time',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Update marker size and scene
    fig_3d_map.update_traces(marker=dict(sizemode='diameter'))
    fig_3d_map.update_layout(
        scene=dict(
            xaxis=dict(range=[lon-1, lon+1], title='Longitude'),
            yaxis=dict(range=[lat-1, lat+1], title='Latitude'),
            zaxis=dict(title='Temperature (°C)'),
        ),
        width=1200,
        height=600
    )
    st.plotly_chart(fig_3d_map, use_container_width=True)

def ml_predictions_page(df, chart_theme):
    """Page for machine learning predictions."""
    st.title("🤖 Machine Learning Weather Predictions")
    st.write("""
    This page shows predictions for the next 5 days using an LSTM neural network.
    The model is trained on historical weather data to forecast temperature, humidity,
    wind speed, and pressure.
    """)
    
    chart_colors = {
        "Vibrant": px.colors.qualitative.Set1,
        "Pastel": px.colors.qualitative.Pastel,
        "Neon": px.colors.qualitative.Bold,
        "Ocean": px.colors.sequential.Blues,
        "Sunset": px.colors.sequential.Sunset
    }
    
    with st.spinner('Training model and generating predictions...'):
        pred_df = predict_weather_attributes(df)
    
    if pred_df is not None:
        # Temperature Prediction
        st.subheader("🌡️ Temperature Prediction")
        fig_temp_pred = go.Figure()
        fig_temp_pred.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['temp'],
            mode='lines',
            name='Historical',
            line=dict(color=chart_colors[chart_theme][0])
        ))
        fig_temp_pred.add_trace(go.Scatter(
            x=pred_df['datetime'],
            y=pred_df['temp'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color=chart_colors[chart_theme][1], dash='dash')
        ))
        fig_temp_pred.update_layout(
            title='LSTM Temperature Prediction',
            xaxis_title='DateTime',
            yaxis_title='Temperature (°C)',
            showlegend=True
        )
        st.plotly_chart(fig_temp_pred, use_container_width=True)
        
        # Humidity Prediction
        st.subheader("💧 Humidity Prediction")
        fig_humidity_pred = go.Figure()
        fig_humidity_pred.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['humidity'],
            mode='lines',
            name='Historical',
            line=dict(color=chart_colors[chart_theme][0])
        ))
        fig_humidity_pred.add_trace(go.Scatter(
            x=pred_df['datetime'],
            y=pred_df['humidity'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color=chart_colors[chart_theme][1], dash='dash')
        ))
        fig_humidity_pred.update_layout(
            title='LSTM Humidity Prediction',
            xaxis_title='DateTime',
            yaxis_title='Humidity (%)',
            showlegend=True
        )
        st.plotly_chart(fig_humidity_pred, use_container_width=True)
        
        # Wind Speed Prediction
        st.subheader("🌬️ Wind Speed Prediction")
        fig_wind_pred = go.Figure()
        fig_wind_pred.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['wind_speed'],
            mode='lines',
            name='Historical',
            line=dict(color=chart_colors[chart_theme][0])
        ))
        fig_wind_pred.add_trace(go.Scatter(
            x=pred_df['datetime'],
            y=pred_df['wind_speed'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color=chart_colors[chart_theme][1], dash='dash')
        ))
        fig_wind_pred.update_layout(
            title='LSTM Wind Speed Prediction',
            xaxis_title='DateTime',
            yaxis_title='Wind Speed (m/s)',
            showlegend=True
        )
        st.plotly_chart(fig_wind_pred, use_container_width=True)
        
        # Pressure Prediction
        st.subheader("⏲️ Pressure Prediction")
        fig_pressure_pred = go.Figure()
        fig_pressure_pred.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['pressure'],
            mode='lines',
            name='Historical',
            line=dict(color=chart_colors[chart_theme][0])
        ))
        fig_pressure_pred.add_trace(go.Scatter(
            x=pred_df['datetime'],
            y=pred_df['pressure'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color=chart_colors[chart_theme][1], dash='dash')
        ))
        fig_pressure_pred.update_layout(
            title='LSTM Pressure Prediction',
            xaxis_title='DateTime',
            yaxis_title='Pressure (hPa)',
            showlegend=True
        )
        st.plotly_chart(fig_pressure_pred, use_container_width=True)
        
        # Download ML Predictions
        st.subheader("📥 Download Predictions")
        st.markdown(get_csv_download_link(pred_df, "ml_weather_predictions.csv"), unsafe_allow_html=True)

def data_export_page(df):
    """Page for data export options."""
    st.title("📥 Data Export")
    
    st.write("""
    Export the weather forecast data in various formats for further analysis.
    """)
    
    # Show raw data
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df)
    
    # Export options
    st.subheader("💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### CSV Format")
        st.markdown(get_csv_download_link(df, "weather_forecast.csv"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### JSON Format")
        json_data = df.to_json(orient='records', date_format='iso')
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="weather_forecast.json",
            mime="application/json"
        )
    
    # Statistics
    st.subheader("📊 Data Statistics")
    st.dataframe(df.describe())

def main():
    # Setup sidebar and get city
    city, chart_theme, bg_color = setup_sidebar()
    
    # Apply custom background color
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
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
    
    if city:
        current_weather, forecast_data = get_weather_data(city)
        
        if current_weather and forecast_data:
            df = process_forecast_data(forecast_data)
            
            # Page navigation
            st.sidebar.title("📄 Page Navigation")
            page = st.sidebar.radio("Go to:", [
                "Dashboard Overview",
                "Weather Analytics",
                "Temperature Forecast",
                "Humidity & Wind",
                "Maps & Visualizations",
                "ML Predictions",
                "Data Export"
            ])
            
            if page == "Dashboard Overview":
                dashboard_page(current_weather, forecast_data, chart_theme)
            elif page == "Weather Analytics":
                weather_analytics_page(df, chart_theme)
            elif page == "Temperature Forecast":
                temperature_page(df, chart_theme)
            elif page == "Humidity & Wind":
                humidity_wind_page(df, chart_theme)
            elif page == "Maps & Visualizations":
                maps_page(current_weather, df)
            elif page == "ML Predictions":
                ml_predictions_page(df, chart_theme)
            elif page == "Data Export":
                data_export_page(df)
        else:
            st.warning("Please enter a valid city name to see weather data.")
    else:
        st.info("🌍 Please select or enter a city to view weather information.")

if __name__ == "__main__":
    main()