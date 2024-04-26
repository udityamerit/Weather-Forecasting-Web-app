import requests
import math
import json
import sys



def get_weather(city):
    api_key = "b3c62ae7f7ad5fc3cb0a7b56cb7cbda6"
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
            sys.exit(1)
    except json.JSONDecodeError as err:
        print(f"Error: Failed to parse response JSON - {err}")
        
    # Extract relevant weather information
    weather_description = data['weather'][0]['description']
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    
    # Convert temperature from Kelvin to Celsius
    temperature = round(temperature - 273.15, 2)
    
    # Print the weather forecast
    print(f"Weather in {city}: {weather_description}")
    print(f"Temperature: {temperature}°C")
    print(f"Humidity: {humidity}")
    print(f"Pressure: {pressure}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <citys>")
        sys.exit(1)
        
    city = sys.argv[1]
    get_weather(city)
    