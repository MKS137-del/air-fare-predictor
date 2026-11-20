# main.py - Flight Price Prediction API (simple version)

import joblib
import pandas as pd
from datetime import date
from fastapi import FastAPI
from pydantic import BaseModel

# Step 1: Create the FastAPI app
app = FastAPI()

# Step 2: Load the saved model (created by train_model.py)
saved = joblib.load("flight_model.pkl")
model = saved["model"]                      # the trained Random Forest
feature_columns = saved["feature_columns"]  # the 33 column names, in training order

# Step 3: Lists of allowed values (used to check the user's input)
airlines = ["Air India", "Akasa Air", "IndiGo", "SpiceJet", "Vistara"]
cabins = ["Economy", "Business"]
times = ["Early Morning", "Morning", "Afternoon", "Evening"]

# Step 4: Routes that exist in our dataset (only these can be predicted)
routes = [
    "Bengaluru to Hyderabad", "Chennai to Hyderabad", "Cochin to Hyderabad",
    "Delhi to Bengaluru", "Delhi to Mumbai", "Hyderabad to Ahmedabad",
    "Hyderabad to Cochin", "Hyderabad to Delhi", "Hyderabad to Mumbai",
    "Kolkata to Delhi", "Mumbai to Bengaluru", "Mumbai to Goa",
]

# Step 5: Duration lookup table (in minutes)
# Key = "source-destination-stops", Value = average duration in our dataset.
# The user does not enter duration, we find it from this table.
durations = {
    "Bengaluru-Hyderabad-0": 77,
    "Bengaluru-Hyderabad-1": 285,
    "Bengaluru-Hyderabad-2": 525,
    "Chennai-Hyderabad-0": 77,
    "Chennai-Hyderabad-1": 284,
    "Chennai-Hyderabad-2": 524,
    "Cochin-Hyderabad-0": 117,
    "Cochin-Hyderabad-1": 325,
    "Cochin-Hyderabad-2": 563,
    "Delhi-Bengaluru-0": 172,
    "Delhi-Bengaluru-1": 379,
    "Delhi-Bengaluru-2": 616,
    "Delhi-Mumbai-0": 133,
    "Delhi-Mumbai-1": 340,
    "Delhi-Mumbai-2": 578,
    "Hyderabad-Ahmedabad-0": 108,
    "Hyderabad-Ahmedabad-1": 315,
    "Hyderabad-Ahmedabad-2": 558,
    "Hyderabad-Cochin-0": 118,
    "Hyderabad-Cochin-1": 325,
    "Hyderabad-Cochin-2": 565,
    "Hyderabad-Delhi-0": 137,
    "Hyderabad-Delhi-1": 345,
    "Hyderabad-Delhi-2": 584,
    "Hyderabad-Mumbai-0": 92,
    "Hyderabad-Mumbai-1": 299,
    "Hyderabad-Mumbai-2": 541,
    "Kolkata-Delhi-0": 148,
    "Kolkata-Delhi-1": 354,
    "Kolkata-Delhi-2": 591,
    "Mumbai-Bengaluru-0": 113,
    "Mumbai-Bengaluru-1": 320,
    "Mumbai-Bengaluru-2": 558,
    "Mumbai-Goa-0": 82,
    "Mumbai-Goa-1": 290,
    "Mumbai-Goa-2": 530,
}

# Step 6: Describe the data the user must send (no duration needed)
class FlightInput(BaseModel):
    airline: str
    source: str
    destination: str
    cabin_class: str
    departure_time: str
    total_stops: int
    days_left: int
    year: int
    month: int
    day: int


# Step 7: Home page
@app.get("/")
def home():
    return {"message": "Flight Price API is running. Go to /docs to test it."}


# Step 8: Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Step 9: Show the routes that are available
@app.get("/routes")
def show_routes():
    return {"available_routes": routes}


# Step 10: Prediction
@app.post("/predict")
def predict(data: FlightInput):

    # --- Check the input ---
    if data.airline not in airlines:
        return {"error": "airline must be one of: " + str(airlines)}
    if data.cabin_class not in cabins:
        return {"error": "cabin_class must be one of: " + str(cabins)}
    if data.departure_time not in times:
        return {"error": "departure_time must be one of: " + str(times)}
    if data.days_left < 1 or data.days_left > 90:
        return {"error": "days_left must be between 1 and 90"}

    # --- Find the duration automatically from the lookup table ---
    key = data.source + "-" + data.destination + "-" + str(data.total_stops)

    if key not in durations:
        return {
            "error": "No data for this route or number of stops (stops must be 0, 1 or 2).",
            "available_routes": routes,
        }

    duration = durations[key]

    # --- Find the day of the week (Monday = 0 ... Sunday = 6) ---
    try:
        day_of_week = date(data.year, data.month, data.day).weekday()
    except ValueError:
        return {"error": "The date is not valid"}

    # In the training data, Friday (4) and Sunday (6) are marked as weekend
    if day_of_week == 4 or day_of_week == 6:
        is_weekend = 1
    else:
        is_weekend = 0

    # --- Make a dictionary with every column set to 0 ---
    row = {}
    for column in feature_columns:
        row[column] = 0

    # --- Fill in the normal number columns ---
    row["total_stops"] = data.total_stops
    row["duration_mins"] = duration
    row["days_left"] = data.days_left
    row["is_weekend"] = is_weekend
    row["flight_year"] = data.year
    row["flight_month"] = data.month
    row["flight_day"] = data.day
    row["flight_dayofweek"] = day_of_week

    # --- Fill in the one-hot columns: set the matching column to 1 ---
    row["airline_" + data.airline] = 1
    row["source_" + data.source] = 1
    row["destination_" + data.destination] = 1
    row["cabin_class_" + data.cabin_class] = 1

    # Departure time columns have longer names, so we use if/elif
    if data.departure_time == "Early Morning":
        row["departure_time_Early Morning (12AM-6AM)"] = 1
    elif data.departure_time == "Morning":
        row["departure_time_Morning (6AM-12PM)"] = 1
    elif data.departure_time == "Afternoon":
        row["departure_time_Afternoon (12PM-6PM)"] = 1
    else:
        row["departure_time_Evening (6PM-12AM)"] = 1

    # --- Put the row in a table (DataFrame) and predict ---
    table = pd.DataFrame([row], columns=feature_columns)
    price = model.predict(table)[0]

    return {
        "predicted_price_inr": round(float(price), 2),
        "duration_used_mins": duration,
    }
