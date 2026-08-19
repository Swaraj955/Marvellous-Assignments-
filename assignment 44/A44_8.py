import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Name':['Amit','Sagar','Puja'],
    'Math':[85,90,78],
    'Science':[92,88,80],
    'English':[75,85,82]
}

df = pd.DataFrame(data)

amit = df[df['Name']=='Amit'].iloc[0]

subjects = ['Math','Science','English']
marks = [amit['Math'], amit['Science'],amit['English']]

plt.plot(subjects,marks,marker ='o')

plt.xlabel("Subjects")
plt.ylabel("Marks")
plt.title("Amit's marks across all Subjects")

plt.show()