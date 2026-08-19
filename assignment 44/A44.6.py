import pandas as pd

data = {
    'Name':['Amit','Sagar','Puja'],
    'Math':[85,90,78],
    'Science':[92,88,80],
    'English':[75,85,82]
}

df = pd.DataFrame(data)

df['Total']= df['Math'] +df['Science']+df['English']

df = df.sort_values(by='Total',ascending=False)

print(df)