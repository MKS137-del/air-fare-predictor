import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load dataset
data_path = os.path.join("ML", "indian_flights_model", "indian_flights_rf_ready.csv")
if not os.path.exists(data_path):
    data_path = "indian_flights_rf_ready.csv"

df = pd.read_csv(data_path)

# 2. Split features and target
X = df.drop(columns='price')
y = df['price']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['cabin_class_Business']
)

# 4. Train Random Forest model with your notebook parameters
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 5. Save the trained model package
model_package = {
    "model": rf,
    "feature_columns": X_train.columns.tolist()
}

joblib.dump(model_package, "flight_model.pkl")
joblib.dump(model_package, "flight_price_model.pkl")

print("Model pipeline saved as flight_model.pkl")
