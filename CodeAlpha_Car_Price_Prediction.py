# CodeAlpha Task 3: Car Price Prediction
# Common dataset columns: Car_Name, Year, Selling_Price, Present_Price,
# Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner.
# Upload your CodeAlpha car-price CSV when prompted.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(filename)
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# Adapt these names if your downloaded dataset uses different headers.
target = "Selling_Price"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found. Check your dataset headers.")

# Basic feature engineering for the common CarDekho dataset
if "Year" in df.columns:
    df["Car_Age"] = pd.Timestamp.today().year - pd.to_numeric(df["Year"], errors="coerce")

# Remove rows without target; ensure target is numeric
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])

# Remove an identifier/name column when present; it is not a numeric measurement
X = df.drop(columns=[target])
for col in ["Car_Name", "name", "Name"]:
    if col in X.columns:
        X = X.drop(columns=col)

y = df[target]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=300, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"\nMAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2 score: {r2:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred, alpha=0.7)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual selling price")
plt.ylabel("Predicted selling price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.show()

# Example prediction: copy one real row from X and modify its values as needed.
example = X_test.iloc[[0]]
print("\nExample input:\n", example)
print("Predicted selling price:", model.predict(example)[0])
