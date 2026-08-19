import pandas as pd

df = pd.DataFrame({
    'Name': ['Sagar', 'Rahul', 'Amit', 'Sneha', 'Priya'],
    'Math': [80, 65, 90, 70, 85],
    'English': [75, 60, 88, 72, 80],
    'Science': [85, 70, 92, 68, 78],
    'Gender' : ['Male', 'Male', 'Male', 'Female', 'Female']

})

df["Average"]=df[["Math","English","Science"]].mean(axis=1)

result = df.groupby("Gender")["Average"].mean()
print("average marks are:")
print(result)