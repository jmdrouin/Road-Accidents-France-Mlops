# Start the streamlit:
# uv run python -m streamlit run src/streamlit/app.py

import streamlit as st
import requests
import os
import pandas as pd
from typing import get_args

from src.models.accident import build_accident_model

Accident = build_accident_model()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001") #8000

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

def ordered_field_values(field_name: str):
    values = field_enum_values(field_name)

    preferred_orders = {
        "season": ["Spring", "Summer", "Autumn", "Winter"],
        "day_of_week": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        "hour_group": [
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
        ],
        "age_group": [
            "Child", "Young_Adult", "Adult", "Middle_Aged", "Senior", "null"
        ]
    }

    preferred = preferred_orders.get(field_name)
    if not preferred:
        return values

    # Keep only values that actually exist in the enum, in the desired order.
    ordered = [v for v in preferred if v in values]

    # Add any unexpected enum values at the end so nothing breaks.
    remaining = [v for v in values if v not in ordered]

    return ordered + remaining

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

    segmented_fields = {
        "season", "day_of_week", "hour_group", "weather_group", "road_group",
        "surface_condition_label", "journey_purpose_label", "age_group",
        "sex_label", "user_category_label",
        "seat_position_label",
        "journey_purpose_label",
        "collision_label",
        "manoeuvre_label",
        "vehicle_group",
        "impact_group",
    }

    for field in fields:
        label = pretty_label(field)

        if field in binary_fields:
            default = bool(field_default(field))

            value = st.toggle(
                label,
                value=default,
                key=f"{field}_toggle",
            )
            data[field] = int(value)

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
            values = ordered_field_values(field)
            default = field_default(field)

            if field in segmented_fields:
                data[field] = st.segmented_control(
                    label,
                    options=values,
                    default=default,
                    selection_mode="single",
                )
            else:
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
    
    st.title("Accident prediction")
    #st.header("")
    
    st.set_page_config(layout="wide")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.header("Accident")
        input_data = build_input_data()
    with right_col:
        st.header("Prediction")
        if st.button("Predict"):
            display_prediction(input_data)

def display_prediction(input_data):
    try:
        accident = Accident(**input_data)

        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=accident.model_dump(),
            timeout=30,
        )
        response.raise_for_status()
        api_result = response.json()

        probabilities = api_result["probabilities"]
        prediction = api_result["prediction"]
        confidence = api_result["confidence"]

        st.subheader("Prediction")
        st.write(f"**{prediction}** ({confidence:.1%})")

        proba_df = pd.DataFrame([probabilities])
        st.dataframe(proba_df, use_container_width=True)

        display_probability_chart_from_dict(probabilities)

    except requests.HTTPError:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        st.error(f"API error: {detail}")
    except Exception as e:
        st.error(str(e))

def display_probability_chart_from_dict(probabilities: dict[str, float]):
    import plotly.graph_objects as go

    color_map = {
        "Killed": "#D32F2F",               # dark red
        "Injured_Hospitalized": "#F57C00", # orange
        "Injured_Slight": "#FBC02D",       # yellow
        "Uninjured": "#388E3C",            # green
    }
    
    desired_order = [
        "Killed",
        "Injured_Hospitalized",
        "Injured_Slight",
        "Uninjured",
    ]

    labels = [label for label in desired_order if label in probabilities]
    values = [float(probabilities[label]) for label in labels]

    fig = go.Figure()

    for label, value in zip(labels, values):
        fig.add_trace(
            go.Bar(
                name=label,
                x=[value],
                y=["Prediction"],
                orientation="h",
                text=[f"{value:.1%}"],
                textposition="inside",
                marker_color=color_map.get(label, "#7f7f7f") #new
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
        yaxis=dict(title=""),
        height=180,
        showlegend=True,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True) #, theme=None

def display_probability_chart(result):
    import plotly.graph_objects as go

    desired_order = [
        "Killed",
        "Injured_Hospitalized",
        "Injured_Slight",
        "Uninjured",
    ]

    row = result.iloc[0]

    labels = []
    values = []

    for label in desired_order:
        col = f"proba_{label}"
        if col in result.columns:
            labels.append(label)
            values.append(float(row[col]))

    fig = go.Figure()

    for label, value in zip(labels, values):
        fig.add_trace(
            go.Bar(
                name=label,
                x=[value],
                y=["Prediction"],
                orientation="h",
                text=[f"{value:.1%}"],
                textposition="inside",
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            title="Probability",
        ),
        yaxis=dict(title=""),
        showlegend=True,
        height=180,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    row = result.iloc[0]
    p_killed = float(row.get("proba_Killed", 0.0))

    st.metric(
        label="Risk of death",
        value=f"{p_killed:.2%}",
    )

    risk_bar(p_killed)

    st.caption(f"≈ 1 in {int(1/p_killed) if p_killed > 0 else '∞'} cases")

    st.plotly_chart(fig, use_container_width=True)

def display_probability_chart_log(result):
    import plotly.graph_objects as go

    desired_order = [
        "Killed",
        "Injured_Hospitalized",
        "Injured_Slight",
        "Uninjured",
    ]

    row = result.iloc[0]

    labels = []
    values = []

    for label in desired_order:
        col = f"proba_{label}"
        if col in result.columns:
            labels.append(label)
            values.append(float(row[col]))

    # avoid log(0)
    epsilon = 1e-6
    log_values = [max(v, epsilon) for v in values]

    fig = go.Figure()

    for label, value, raw_value in zip(labels, log_values, values):
        fig.add_trace(
            go.Bar(
                name=label,
                x=[value],
                y=["Prediction"],
                orientation="h",
                text=[f"{raw_value:.2%}"],
                textposition="inside",
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis=dict(
            type="log",
            title="Probability (log scale)",
        ),
        yaxis=dict(title=""),
        height=180,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

def risk_bar(p):
    # Color thresholds (tune these)
    if p < 0.01:
        color = "#2ca02c"   # green
    elif p < 0.05:
        color = "#ff7f0e"   # orange
    else:
        color = "#d62728"   # red

    st.markdown(
        f"""
        <div style="
            background-color: #eee;
            border-radius: 8px;
            height: 20px;
            width: 100%;
        ">
            <div style="
                background-color: {color};
                width: {p*5000}%;
                height: 100%;
                border-radius: 8px;
                text-align: right;
                padding-right: 6px;
                color: white;
                font-size: 12px;
                line-height: 20px;
            ">
                {p:.2%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _display_probability_chart(result):
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