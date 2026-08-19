import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    'Name': ['Sagar', 'Rahul', 'Amit', 'Sneha', 'Priya'],
    'Math': [80, 65, 90, 70, 85],
    'English': [75, 60, 88, 72, 80],
    'Science': [85, 70, 92, 68, 78]
})

scaler = MinMaxScaler()
df['Math'] = scaler.fit_transform(df[['Math']])

print(df)  