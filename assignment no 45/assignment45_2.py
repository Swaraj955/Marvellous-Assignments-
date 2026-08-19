import pandas as pd

df = pd.DataFrame({
    'Name': ['Sagar', 'Rahul', 'Amit', 'Sneha', 'Priya'],
    'Math': [80, 65, 90, 70, 85],
    'English': [75, 60, 88, 72, 80],
    'Science': [85, 70, 92, 68, 78]
})

df['Gender'] = ['Male', 'Male', 'Male', 'Female', 'Female']

df = pd.get_dummies(df, columns=['Gender'])

print(df)