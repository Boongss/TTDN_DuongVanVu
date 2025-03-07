from flask import Flask, jsonify 
import pandas as pd 
import os

app = Flask(__name__)

RAW_DATA_PATH = '../data/weather_raw.csv'

@app.route('/fetch_weather', methods=['GET'])
def fetch_weather():
    if os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH)
        data = df.to_dict(orient='records')
        return jsonify(data)
    else:
        return jsonify({'error': 'No data available. Please run fetch_weather_data.py first.'})

if __name__ == '__main__':
    app.run(debug=True)