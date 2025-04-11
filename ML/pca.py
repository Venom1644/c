# feature transformantion

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (we'll ignore labels for PCA)

# Optional: convert to DataFrame for clarity
df = pd.DataFrame(X, columns=iris.feature_names)
print("Original Feature Data:")
print(df.head())

# Step 1: Standardize the features (very important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X_scaled)

# Step 3: Convert to DataFrame
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

print("\nTransformed Feature Data (PCA):")
print(df_pca.head())

# Optional: Add target back for visualization or further use
df_pca['target'] = y
