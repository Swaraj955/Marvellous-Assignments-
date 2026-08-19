import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Name': ['Sagar', 'Rahul', 'Amit', 'Sneha', 'Priya'],
    'Math': [80, 65, 90, 70, 85],
    'English': [75, 60, 88, 72, 80],
    'Science': [85, 70, 92, 68, 78]
})

sagar = df[df['Name'] == 'Sagar'].iloc[0]

marks = [sagar['Math'], sagar['English'], sagar['Science']]
subjects = ['Math', 'English', 'Science']

plt.pie(marks, labels=subjects, autopct='%1.1f%%')
plt.title("Sagar's Subject Marks")
plt.show()