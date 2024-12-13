import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 讀取數據
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_labels = pd.read_csv('train_label.csv')

# 顯示數據信息
print("訓練數據形狀:", train_data.shape)
print("測試數據形狀:", test_data.shape)
print("訓練標籤形狀:", train_labels.shape)

print("\n訓練數據列名:")
print(train_data.columns)

print("\n訓練數據描述統計:")
print(train_data.describe())

print("\n訓練數據前幾行:")
print(train_data.head())

print("\n訓練標籤列名:")
print(train_labels.columns)

print("\n標籤分布:")
print(train_labels['label'].value_counts(normalize=True))

# 合併訓練數據和標籤
train_data = pd.merge(train_data, train_labels, on='mac_hash')

# 將 'created_time' 轉換為 datetime 對象
train_data['created_time'] = pd.to_datetime(train_data['created_time'])
test_data['created_time'] = pd.to_datetime(test_data['created_time'])

# 特徵工程
def extract_features(df):
    df['hour'] = df['created_time'].dt.hour
    df['day'] = df['created_time'].dt.day
    df['month'] = df['created_time'].dt.month
    df['dayofweek'] = df['created_time'].dt.dayofweek
    return df

train_data = extract_features(train_data)
test_data = extract_features(test_data)

# 對 'sniffer_loc' 進行標籤編碼
le = LabelEncoder()
train_data['sniffer_loc'] = le.fit_transform(train_data['sniffer_loc'])
test_data['sniffer_loc'] = le.transform(test_data['sniffer_loc'])

# 選擇特徵
features = ['sniffer_loc', 'hour', 'day', 'month', 'dayofweek']
X = train_data[features]
y = train_data['label']

# 分割訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練隨機森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 在驗證集上評估模型
val_predictions = rf_model.predict(X_val)
print("\n驗證集準確率:", accuracy_score(y_val, val_predictions))
print("\n分類報告:")
print(classification_report(y_val, val_predictions))

# 對測試集進行預測
test_features = test_data[features]
test_predictions = rf_model.predict(test_features)

# 創建提交文件
submission = pd.DataFrame({
    'mac_hash': test_data['mac_hash'],
    'label': test_predictions
})
submission.to_csv('submit.csv', index=False)
print("\n提交文件已生成: submit.csv")
