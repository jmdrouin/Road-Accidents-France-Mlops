# Start the streamlit:
# uv run python -m streamlit run src/streamlit/app.py

import streamlit as st
import pandas as pd
from typing import get_args

from src.models.accident import build_accident_model
from src.models.predict_model import predict_dataframe, load_artifact
from src.util import last_file_in_folder

Accident = build_accident_model()

PREDICT_COLUMNS = [
    "timestamp", "collision_label", "is_weekend", "season",
    "surface_condition_label", "manoeuvre_label", "sex_label",
    "user_category_label", "seat_position_label", "journey_purpose_label",
    "is_holiday", "age", "age_group", "seatbelt_used", "helmet_used",
    "any_protection_used", "protection_effective", "vehicle_group",
    "impact_group", "motorcycle_side_impact", "is_night", "is_urban",
    "lane_width", "road_group", "weather_group", "day_of_week",
    "hour_group"
]


def field_enum_values(field_name: str):
    annotation = Accident.model_fields[field_name].annotation
    return list(get_args(annotation))


def field_default(field_name: str):
    return Accident.model_fields[field_name].default


def build_input_data():
    data = {}

    st.subheader("Numeric fields")
    data["age"] = st.number_input(
        "age",
        min_value=0,
        max_value=106,
        value=field_default("age") or 30,
    )
    data["lane_width"] = st.number_input(
        "lane_width",
        min_value=0.5,
        max_value=200.0,
        value=float(field_default("lane_width") or 3.0),
    )

    st.subheader("Categorical fields")
    categorical_fields = [
        "collision_label",
        "season",
        "surface_condition_label",
        "manoeuvre_label",
        "sex_label",
        "user_category_label",
        "seat_position_label",
        "journey_purpose_label",
        "age_group",
        "vehicle_group",
        "impact_group",
        "road_group",
        "weather_group",
        "day_of_week",
        "hour_group",
    ]

    for field in categorical_fields:
        values = field_enum_values(field)
        default = field_default(field)
        index = values.index(default)
        data[field] = st.selectbox(field, values, index=index)

    st.subheader("Binary fields")
    binary_fields = [
        "is_weekend",
        "is_holiday",
        "seatbelt_used",
        "helmet_used",
        "any_protection_used",
        "protection_effective",
        "motorcycle_side_impact",
        "is_night",
        "is_urban",
    ]

    for field in binary_fields:
        default = bool(field_default(field))
        label = field.replace("_", " ").capitalize()
        data[field] = 1 if st.checkbox(label, value=default) else 0

    return data


def predict_demo():
    st.title("Accident prediction")

    model_file = last_file_in_folder("models", "model_*.pkl")
    artifact = load_artifact(model_file)

    input_data = build_input_data()

    if st.button("Predict"):
        try:
            accident = Accident(**input_data)
            row = accident.model_dump()
            row["timestamp"] = None

            df = pd.DataFrame([row])[PREDICT_COLUMNS]
            result = predict_dataframe(df, artifact).copy()
            
            # Prevent bug where streamlit can't display "largeUtf8":
            result = result.astype(object)

            probas = [c for c in result.columns if c.startswith("proba_")]

            st.subheader("Prediction")
            st.dataframe(result[probas])

            row = result.iloc[0]
            best = max(probas, key=lambda c: row[c])
            label = best.replace("proba_", "")
            confidence = row[best]
            st.subheader(f"{label} (p={100*confidence}%)")

            display_probability_chart(result[probas])

        except Exception as e:
            st.error(str(e))

def display_probability_chart(result):
    import plotly.graph_objects as go

    probas = [c for c in result.columns if c.startswith("proba_")]
    row = result.iloc[0]

    labels = [c.replace("proba_", "") for c in probas]
    values = [float(row[c]) for c in probas]

    # optional: sort from highest to lowest
    # pairs = sorted(zip(labels, values), key=lambda x: x[1])
    # labels = [p[0] for p in pairs]
    # values = [p[1] for p in pairs]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(title=""),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    predict_demo()