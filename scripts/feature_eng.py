import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import preprocess_data

# Load data
data = pd.read_csv("data/train.csv")

# Preprocess data
data = preprocess_data(data)

# Split into X (features) and y (target)
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Train simple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Get feature importance (absolute values)
feature_importance = pd.Series(np.abs(model.coef_), index=X.columns)

# Sort features by importance
feature_importance = feature_importance.sort_values(ascending=True)

# Display least important features
print("Least important features:\n", feature_importance.head(10))
