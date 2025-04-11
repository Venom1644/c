
dict_data={
    'customer_id':[1,-1,3,4,5],
    'items_bought':[10,24,44,-1,34],
    'price':[566,899,-1,357,852]}

import pandas as pd
df_dict_data= pd.DataFrame(dict_data)

from sklearn.impute import KNNImputer
knn= KNNImputer(n_neighbors=4, missing_values=-1)
knn_data= knn.fit_transform(df_dict_data)

res = pd.DataFrame(knn_data, columns=df_dict_data.columns)
print(res)