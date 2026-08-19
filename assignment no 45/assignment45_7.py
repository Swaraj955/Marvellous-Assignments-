import pandas as pd

df = pd.DataFrame({
    'Name': ['Sagar', 'Rahul', 'Amit', 'Sneha', 'Priya'],
    'Math': [80, 65, 90, 70, 85],
    'English': [75, 60, 88, 72, 80],
    'Science': [85, 70, 92, 68, 78]
})

df['Total'] = df[['Math', 'English', 'Science']].sum(axis=1)

df['Status'] = df['Total'].apply(lambda x: 'Pass' if x >= 250 else 'Fail')

df.to_csv('final_students.csv', index=False)

print("DataFrame exported successfully.")