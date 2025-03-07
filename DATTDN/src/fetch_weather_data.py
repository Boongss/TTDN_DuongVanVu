import requests
import pandas as pd 
import os
from datetime import datetime, timedelta

CITY = 'Hanoi'
SEARCH_URL = f'https://www.metaweather.com/api/location/search/?query={CITY}'
WEATHER_URL = 'https://www.metaweather.com/api/location/'
RAW_DATA_PATH = '../data/weather_raw.csv'

def fetch_weather_data():
    # Xóa file cũ nếu tồn tại
    if os.path.exists(RAW_DATA_PATH):
        os.remove(RAW_DATA_PATH)
    
    search_response = requests.get(SEARCH_URL)
    search_data = search_response.json()
    if search_data:
        woeid = search_data[0]['woeid']
        weather_response = requests.get(f'{WEATHER_URL}{woeid}/')
        weather_data = weather_response.json()
        df = pd.json_normalize(weather_data['consolidated_weather'])
        
        # Lọc dữ liệu theo các mốc thời gian cụ thể
        df['applicable_date'] = pd.to_datetime(df['applicable_date'])
        df = df[df['applicable_date'] >= datetime.now() - timedelta(days=7)]
        df['hour'] = df['applicable_date'].dt.hour
        df = df[df['hour'].isin([6, 9, 12, 15, 18, 21, 0, 3])]
        
        df.to_csv(RAW_DATA_PATH, index=False)
        return df.to_dict(orient='records')
    else:
        return {'error': f'City {CITY} not found.'}

if __name__ == '__main__':
    fetch_weather_data()