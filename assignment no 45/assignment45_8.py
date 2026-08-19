import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Name': ['Sagar', 'Rahul', 'Amit', 'Sneha', 'Priya'],
    'Math': [80, 65, 90, 70, 85]
})

plt.hist(df['Math'], bins=5)
plt.xlabel('Math Marks')
plt.ylabel('Number of Students')
plt.title('Histogram of Math Marks')
plt.show()