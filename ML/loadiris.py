import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()

df= pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)