import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Name': ['Sagar', 'Rahul', 'Amit', 'Sneha', 'Priya'],
    'English': [75, 60, 88, 72, 80]
})

plt.boxplot(df['English'])
plt.ylabel('English Marks')
plt.title('Boxplot of English Marks')
plt.show()