import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Kiểm tra và cài đặt thư viện nếu chưa có
def install_missing_libraries():
    try:
        import pip
        required_packages = ['torch', 'numpy', 'pandas', 'openpyxl']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                print(f"📦 Cài đặt thư viện {package}...")
                os.system(f"python -m pip install {package}")
        print("✅ Tất cả thư viện đã được cài đặt!")
    except Exception as e:
        print(f"❌ Lỗi khi cài đặt thư viện: {e}")

install_missing_libraries()

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class WeatherRecommender:
    def __init__(self, num_actions=9, learning_rate=0.001, discount_factor=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, num_episodes=500, batch_size=32, memory_size=10000):
        self.data_path = "./data/data.xlsx"  
        self.output_path = "./data/"
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.accuracy_data = []
        
        self.actions = [
            "Mang ô", "Mặc áo khoác", "Bôi kem chống nắng", "Ở nhà", "Không khuyến nghị gì",
            "Uống đủ nước", "Đeo kính râm", "Hạn chế ra ngoài", "Mặc quần áo thoáng mát"
        ]
        self.df = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file: {self.data_path}. Kiểm tra lại đường dẫn!")
        
        # Đọc dữ liệu và chuyển đổi tất cả thành kiểu số nếu có thể
        self.df = pd.read_excel(self.data_path, dtype=str)  # Đọc tất cả dưới dạng chuỗi trước
        
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')  # Chuyển đổi sang số, nếu lỗi thì thành NaN
        
        self.df.fillna(method='ffill', inplace=True)  # Điền giá trị thiếu bằng forward fill
        self.df.fillna(method='bfill', inplace=True)  # Điền giá trị thiếu bằng backward fill
        
        if 'temp' in self.df.columns:
            self.df["temp_C"] = (self.df["temp"] - 32) * 5/9  # Chuyển đổi nhiệt độ từ F sang C
            self.df.drop(columns=["tempmax", "tempmin", "temp"], inplace=True, errors='ignore')
        
        self.state_dim = len(self.df.columns) - 1
        print(f"✅ Dữ liệu đã tải thành công! Tổng số dòng: {self.df.shape[0]}, Số cột: {self.df.shape[1]}")

    def normalize_data(self):
        for col in ["temp_C", "humidity", "precipprob", "windgust", "cloudcover", "uvindex"]:
            self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())

    def preprocess_data(self):
        self.load_data()
        self.normalize_data()

    def initialize_model(self):
        self.model = DeepQNetwork(self.state_dim, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def train_dqn(self):
        print("🚀 Bắt đầu huấn luyện mô hình...")
        self.initialize_model()
        
        for episode in range(self.num_episodes):
            correct_predictions = 0
            for i in range(len(self.df)):
                state = torch.tensor(self.df.iloc[i, :-1].values, dtype=torch.float32).to(self.device)
                action = np.random.choice(self.num_actions) if np.random.rand() < self.epsilon else torch.argmax(self.model(state)).item()
                reward = 3 if action == self.get_correct_action(i) else -3
                next_state = torch.tensor(self.df.iloc[min(i+1, len(self.df)-1), :-1].values, dtype=torch.float32).to(self.device)
                done = i == len(self.df) - 1
                
                if action == self.get_correct_action(i):
                    correct_predictions += 1
            
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.accuracy_data.append({"Episode": episode + 1, "Accuracy": correct_predictions / len(self.df)})
        
        print("✅ Training completed, saving results...")
        self.save_results()

    def save_results(self):
        try:
            save_path = self.output_path
            print("📂 Saving processed data...")
            self.df.to_excel(os.path.join(save_path, "processed_data.xlsx"), index=False)
            
            print("📂 Saving accuracy data...")
            accuracy_df = pd.DataFrame(self.accuracy_data)
            accuracy_df.to_excel(os.path.join(save_path, "accuracy_per_episode.xlsx"), index=False)
            
            print("✅ Files saved successfully!")
        except Exception as e:
            print(f"❌ Lỗi khi lưu file: {e}")

# Cách sử dụng:
if __name__ == "__main__":
    recommender = WeatherRecommender(num_episodes=500, batch_size=32)  # Tối ưu cho máy tính cá nhân
    recommender.preprocess_data()
    recommender.train_dqn()
    print("✅ Hoàn thành! Kiểm tra file đầu ra.")
