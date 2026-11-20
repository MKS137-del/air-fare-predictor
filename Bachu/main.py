from pathlib import Path
from datetime import datetime, date
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Indian Domestic Airfare Predictor API",
    description="Predict domestic Indian flight airfares across all operating airlines with automatic booking lead time and weekend surcharge calculation."
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "app" / "rf_airfare_model.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "rf_airfare_model.pkl"

# Load the serialized Random Forest pipeline once at startup
model = joblib.load(MODEL_PATH)


class FlightInput(BaseModel):
    travel_date: str = Field(
        ...,
        description="Travelling date in YYYY-MM-DD format",
        json_schema_extra={"example": "2026-11-28"}
    )
    source: str = Field(
        ...,
        description="Origin city (e.g., Delhi, Mumbai, Bengaluru, Hyderabad, Kolkata, Chennai, Cochin)",
        json_schema_extra={"example": "Delhi"}
    )
    destination: str = Field(
        ...,
        description="Destination city (e.g., Mumbai, Delhi, Bengaluru, Hyderabad, Ahmedabad, Goa, Cochin)",
        json_schema_extra={"example": "Mumbai"}
    )
    cabin_class: str = Field(
        default="Economy",
        description="Cabin class ('Economy' or 'Business')",
        json_schema_extra={"example": "Economy"}
    )
    departure_time: str = Field(
        default="Morning (6AM-12PM)",
        description="Departure window ('Early Morning (12AM-6AM)', 'Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)')",
        json_schema_extra={"example": "Morning (6AM-12PM)"}
    )
    total_stops: str = Field(
        default="non-stop",
        description="Flight stops ('non-stop', '1-stop', '2-stops')",
        json_schema_extra={"example": "non-stop"}
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "travel_date": "2026-11-28",
                "source": "Delhi",
                "destination": "Mumbai",
                "cabin_class": "Economy",
                "departure_time": "Morning (6AM-12PM)",
                "total_stops": "non-stop"
            }
        }
    }


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Airfare prediction API is running with automated date and multi-airline estimation."
    }


@app.post("/predict")
def predict_fare(flight: FlightInput):
    # 1. Parse and validate travel date
    try:
        travel_dt = datetime.strptime(flight.travel_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid travel_date format. Please provide a valid date formatted as YYYY-MM-DD."
        )

    # 2. Automated calendar calculations
    current_date = date.today()
    raw_days_left = (travel_dt - current_date).days
    days_left = max(1, min(90, raw_days_left))
    journey_month = travel_dt.month
    
    # In this dataset, Friday (4) and Sunday (6) capture peak weekend travel surcharges
    is_weekend = 1 if travel_dt.weekday() in [4, 6] else 0

    # 3. Determine available airlines based on cabin class
    # (Only Air India and Vistara offer Business Class cabins; all 5 airlines operate Economy)
    if flight.cabin_class.strip().title() == "Business":
        available_airlines = ["Air India", "Vistara"]
    else:
        available_airlines = ["Air India", "Akasa Air", "IndiGo", "SpiceJet", "Vistara"]

    # 4. Predict fare across each operating airline
    airline_offers = []
    for airline in available_airlines:
        flight_record = pd.DataFrame([{
            "airline": airline,
            "source": flight.source,
            "destination": flight.destination,
            "cabin_class": flight.cabin_class,
            "departure_time": flight.departure_time,
            "total_stops": flight.total_stops,
            "days_left": days_left,
            "journey_month": journey_month,
            "is_weekend": is_weekend
        }])

        pred_fare = float(model.predict(flight_record)[0])
        tier = "Full-Service Carrier" if airline in ["Air India", "Vistara"] else "Low-Cost Carrier"

        airline_offers.append({
            "airline": airline,
            "carrier_type": tier,
            "predicted_fare_inr": round(pred_fare, 2),
            "formatted_fare": f"₹{pred_fare:,.2f}"
        })

    # Sort offers by lowest fare
    airline_offers = sorted(airline_offers, key=lambda x: x["predicted_fare_inr"])
    for rank, offer in enumerate(airline_offers, 1):
        offer["rank"] = rank

    return {
        "status": "success",
        "flight_summary": {
            "route": f"{flight.source} -> {flight.destination}",
            "travel_date": flight.travel_date,
            "day_of_week": travel_dt.strftime("%A"),
            "current_date": str(current_date),
            "booking_lead_days": raw_days_left,
            "journey_month": journey_month,
            "is_weekend_surge": bool(is_weekend),
            "cabin_class": flight.cabin_class,
            "departure_time": flight.departure_time,
            "total_stops": flight.total_stops
        },
        "best_value_airline": airline_offers[0]["airline"],
        "lowest_fare_formatted": airline_offers[0]["formatted_fare"],
        "all_airline_offers": airline_offers
    }
