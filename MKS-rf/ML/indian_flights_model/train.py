import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("indian_flights_rf_ready.csv")

# 2. Split into X (features) and y (target)
X = df.drop(columns='price')
y = df['price']

# 3. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['cabin_class_Business']
)

# 4. Train the Random Forest Regressor
print("Training model...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 5. Save the model to a pkl file
model_package = {
    "model": rf,
    "feature_columns": X_train.columns.tolist()
}

# Save in current folder
joblib.dump(model_package, "flight_price_model.pkl")

# Save a copy in main folder for FastAPI
joblib.dump(model_package, "../../flight_price_model.pkl")

print("Model trained and saved to flight_price_model.pkl successfully!")
