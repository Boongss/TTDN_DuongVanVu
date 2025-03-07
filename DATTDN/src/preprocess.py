import pandas as pd 
import os

RAW_DATA_PATH = '../data/weather_raw.csv'
PROCESSED_DATA_PATH = '../data/weather_processed.csv'

def preprocess_data(input_file, output_file):
    # Xóa file cũ nếu tồn tại
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(input_file)
    
    # Thực hiện các bước xử lý dữ liệu
    # Ví dụ: chuyển đổi nhiệt độ từ Kelvin sang Celsius
    df['the_temp_celsius'] = df['the_temp']
    
    # Lưu dữ liệu đã xử lý vào file CSV mới
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)