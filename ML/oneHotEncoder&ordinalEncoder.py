# # progg  to encode both oordinal nd nominal data in datasets



import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Sample data
emp = [
    ["pranav", "Mid", "cs"],
    ["uuday", "Intern", "cs"],
    ["sam", "Senior", "cs"]
]

# Create DataFrame
df_emp = pd.DataFrame(emp, columns=['name', 'exp_level', 'department'])
print("Original DataFrame:")
print(df_emp)

# Define column transformer
emp_ct = ColumnTransformer(
    transformers=[
        ('exp', OrdinalEncoder(), ['exp_level']),       # Ordinal encoding for experience level
        ('dept', OneHotEncoder(), ['department'])       # One-hot encoding for department
    ],
    remainder='passthrough'  # Keep other columns like 'name'
)

# Apply transformation
employee = emp_ct.fit_transform(df_emp)

# Create new DataFrame with proper column names
df_employee = pd.DataFrame(employee, columns=emp_ct.get_feature_names_out())
print("\nTransformed DataFrame:")
print(df_employee)
