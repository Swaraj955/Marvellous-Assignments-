import pandas as pd

df = pd.DataFrame({
    'Name': ['Sagar', 'Rahul', 'Amit', 'Sneha', 'Priya'],
    'Math': [80, 65, 90, 70, 85],
    'English': [75, 60, 88, 72, 80],
    'Science': [85, 70, 92, 68, 78],
    'Gender' : ['Male', 'Male', 'Male', 'Female', 'Female']

})

df['total'] = df[['Math','English','Science']].sum(axis=1)
df['status'] = df['total'].apply(lambda x:'Pass' if x>=250 else 'Fail')

print(df)