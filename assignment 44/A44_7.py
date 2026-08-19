import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Name':['Amit','Sagar','Puja'],
    'Math':[85,90,78],
    'Science':[92,88,80],
    'English':[75,85,82]
}

df = pd.DataFrame(data)

df['Total']= df['Math'] +df['Science']+df['English']

plt.bar(df['Name'],df['Total'])
plt.xlabel("student Name")
plt.ylabel("Total marks")
plt.title("Student Name vd Total Marks")

plt.show()