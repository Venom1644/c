import pandas as pd 

dfexcel= pd.read_excel("a1.xlsx")
# print(dfexcel)

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

dfexcel['education_encoded']= le.fit_transform(dfexcel['marks'])
print(dfexcel)
