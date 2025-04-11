import pandas as pd
dfexcel =pd.read_excel("a1.xlsx")

from sklearn.impute import SimpleImputer
imputer= SimpleImputer(strategy="mean")
val= imputer.fit_transform(dfexcel)
res=pd.DataFrame(val,columns=dfexcel.columns)
print(res)