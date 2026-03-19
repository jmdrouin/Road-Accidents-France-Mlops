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


def pretty_label(field: str) -> str:
    labels = {
        "is_urban": "Urban area",
        "weather_group": "Weather",
        "road_group": "Road type",
        "surface_condition_label": "Surface condition",
        "lane_width": "Lane width",
        "age_group": "Age group",
        "sex_label": "Sex",
        "user_category_label": "User category",
        "seat_position_label": "Seat position",
        "journey_purpose_label": "Journey purpose",
        "day_of_week": "Day of week",
        "hour_group": "Hour group",
        "collision_label": "Collision type",
        "manoeuvre_label": "Manoeuvre",
        "vehicle_group": "Vehicle group",
        "impact_group": "Impact group",
        "seatbelt_used": "Seatbelt used",
        "helmet_used": "Helmet used",
        "any_protection_used": "Any protection used",
        "protection_effective": "Protection effective",
        "motorcycle_side_impact": "Motorcycle side impact",
    }
    return labels.get(field, field.replace("_", " ").title())

def render_fields(fields: list[str], data: dict):
    binary_fields = {
        "is_weekend",
        "is_holiday",
        "is_night",
        "seatbelt_used",
        "helmet_used",
        "any_protection_used",
        "protection_effective",
        "motorcycle_side_impact",
        "is_urban",
    }

    numeric_fields = {
        "age",
        "lane_width",
    }

    for field in fields:
        label = pretty_label(field)

        if field in binary_fields:
            default = bool(field_default(field))
            data[field] = 1 if st.checkbox(label, value=default) else 0

        elif field in numeric_fields:
            default = field_default(field)

            if field == "age":
                data[field] = st.number_input(
                    label,
                    min_value=0,
                    max_value=106,
                    value=int(default or 30),
                )
            elif field == "lane_width":
                data[field] = st.number_input(
                    label,
                    min_value=0.5,
                    max_value=200.0,
                    value=float(default or 3.0),
                )

        else:
            values = field_enum_values(field)
            default = field_default(field)
            index = values.index(default)
            data[field] = st.selectbox(label, values, index=index)

def build_input_data():
    data = {}

    st.divider()
    st.subheader("TIME")
    render_fields(
        [
            "season",
            "day_of_week",
            "is_weekend",
            "is_holiday",
            "is_night",
            "hour_group",
        ],
        data,
    )

    st.divider()
    st.subheader("CONDITIONS")
    render_fields(
        [
            "is_urban",
            "weather_group",
            "road_group",
            "surface_condition_label",
            "lane_width",
        ],
        data,
    )

    st.divider()
    st.subheader("USER")
    render_fields(
        [
            "age",
            "age_group",
            "sex_label",
            "user_category_label",
            "seat_position_label",
            "journey_purpose_label",
        ],
        data,
    )

    st.divider()
    st.subheader("ACCIDENT")
    render_fields(
        [
            "collision_label",
            "manoeuvre_label",
            "vehicle_group",
            "impact_group",
            "seatbelt_used",
            "helmet_used",
            "any_protection_used",
            "protection_effective",
            "motorcycle_side_impact",
        ],
        data,
    )

    return data

def predict_demo():
    st.header("")
    st.set_page_config(layout="wide")

    # TODO: Memoize those
    model_file = last_file_in_folder("models", "model_*.pkl")
    artifact = load_artifact(model_file)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.header("Accident")
        input_data = build_input_data()
    with right_col:
        st.header("Prediction")
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