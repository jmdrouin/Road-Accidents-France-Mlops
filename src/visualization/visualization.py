"""
Methods for visualization copied from "Step 1..." jupyter notebook.
They take cleaned up dataframes as inputs.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def visualize_overview(
    caract: pd.DataFrame,
    places: pd.DataFrame
):
    #1
    # Accidents per Year (Line Plot)
    #This shows long‑term accident trends across 2005–2016.

    # Group by year and count accidents
    accidents_per_year = caract.groupby("year").size()

    # Plot
    plt.figure(figsize=(6,4))
    accidents_per_year.plot(kind="line", marker="o", color="steelblue")
    plt.title("Accidents per Year in France")
    plt.xlabel("Year")
    plt.ylabel("Number of Accidents")
    plt.grid(True)
    plt.show()

    # a decline over time, reflecting road safety improvements.

    #2
    # Accidents by Time of Day (Bar Chart)
    # This highlights daily risk patterns

    accidents_timeofday = caract["time_of_day"].value_counts()

    plt.figure(figsize=(4,3))
    accidents_timeofday.plot(kind="bar", color="coral")
    plt.title("Accidents by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Accidents")
    plt.show()

    # Evening and Afternoon peaks (rush hour)

    #3
    # Spatial Distribution (Scatter Plot)
    # This confirms clustering inside France

    plt.figure(figsize=(10,10))
    caract[caract["valid_geo"]].plot.scatter(x="longitude_num", y="latitude_num", alpha=0.1, s=1, color="blue")

    plt.title("Accident Locations Across France")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.show()

    # It shows dense clusters around Paris, Lyon, Marseille, Lille

    #4
    # Accidents by Weather or Lighting (Bar Chart)
    # This shows environmental influence

    plt.figure(figsize=(6,4))
    caract["weather_label"].value_counts().plot(kind="bar", color="seagreen")
    plt.title("Accidents by Weather Condition")
    plt.xlabel("Weather")
    plt.ylabel("Number of Accidents")
    plt.show()

    plt.figure(figsize=(6,4))
    caract["lighting_label"].value_counts().plot(kind="bar", color="purple")
    plt.title("Accidents by Lighting Condition")
    plt.xlabel("Lighting")
    plt.ylabel("Number of Accidents")
    plt.show()

    # Accident count by road category
    places["road_category_label"].value_counts().plot(
        kind="bar",
        figsize=(6,4),
        color="skyblue",
        edgecolor="black"
    )

    plt.title("Accident Count by Road Category")
    plt.xlabel("Road Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    # Accident count by traffic regime
    places["traffic_regime_label"].value_counts().plot(
        kind="bar",
        figsize=(6,4),
        color="salmon",
        edgecolor="black"
    )

    plt.title("Accident Count by Traffic Regime")
    plt.xlabel("Traffic Regime")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    # Bidirectional roads are clearly the most dangerous — likely due to lack of separation and unpredictable maneuvers.
    # One-way and separated roads are safer, supporting infrastructure investment in controlled traffic flow.
    # Unknown values should be monitored but don’t dominate.

    # 3 
    # Accident Count by Surface Condition
    places["surface_condition_label"].value_counts().plot(
        kind="bar",
        figsize=(6,4),
        color="mediumseagreen",
        edgecolor="black"
    )

    plt.title("Accident Count by Surface Condition")
    plt.xlabel("Surface Condition")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    # Most accidents happen on dry, normal roads, not extreme conditions.
    # This reinforces the idea that driver behavior and road design are bigger contributors than weather.
    # Wet roads still pose a significant risk — especially in transitional weather.

    #4
    # Accident count by road profile

    places["road_profile_label"].value_counts().plot(
        kind="bar",
        figsize=(6,4),
        color="goldenrod",
        edgecolor="black"
    )

    plt.title("Accident Count by Road Profile")
    plt.xlabel("Road Profile")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

def visualize_injury_severity(users: pd.DataFrame):
    # VISUALIZATION

    #1 
    # Injury Severity by User Category
    # Shows how outcomes differ between drivers, passengers, and pedestrians

    # Cross-analysis: Injury Severity vs User Category
    cross_tab = users.groupby(["user_category_label", "injury_severity_label"]).size().unstack(fill_value=0)

    cross_tab.plot(kind="bar",figsize=(8,4),colormap="tab20",edgecolor="black")

    plt.title("Injury Severity by User Category")
    plt.xlabel("User Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Injury Severity")
    plt.show()

    #  Drivers are the most exposed group, but also have the highest chance of walking away uninjured.
    # Pedestrians show a concerning proportion of serious injuries and fatalities — even with fewer total cases.

    #2
    # Cross-analysis: Injury Severity vs Safety Equipment Usage
    cross_tab = users.groupby(["safety_equipment_usage", "injury_severity_label"]).size().unstack(fill_value=0)

    cross_tab.plot(
        kind="bar",
        stacked=True,
        figsize=(8,4),
        colormap="tab20",
        edgecolor="black"
    )

    plt.title("Injury Severity vs Safety Equipment Usage")
    plt.xlabel("Safety Equipment Usage")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Injury Severity")
    plt.show()

    #- Used equipment is clearly protective — fewer severe injuries and deaths.
    # Not used and Undetermined groups show higher risk — these are critical for safety policy.

    #3
    # Cross-analysis: Injury Severity vs Seat Position
    cross_tab = users.groupby(["seat_position_label", "injury_severity_label"]).size().unstack(fill_value=0)

    cross_tab.plot(kind="bar",figsize=(12,6),colormap="tab20",edgecolor="black")

    plt.title("Injury Severity by Seat Position")
    plt.xlabel("Seat Position")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Injury Severity")
    plt.tight_layout()
    plt.show()

    # Driver seat is the most vulnerable, but also has the highest chance of walking away uninjured.
    # Rear seats show a safer profile, supporting the idea that rear passengers are less exposed.
    #Outside vehicle and Unknown categories need careful handling in modeling — they may skew results if not filtered properly.

    #4
    # Cross-analysis: Injury Severity vs Journey Purpose

    cross_tab = users.groupby(["journey_purpose_label", "injury_severity_label"]).size().unstack(fill_value=0)

    cross_tab.plot( kind="bar",stacked=True, figsize=(10,6), colormap="tab20", edgecolor="black")

    plt.title("Injury Severity by Journey Purpose")
    plt.xlabel("Journey Purpose")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Injury Severity")
    plt.tight_layout()
    plt.show()

def visualize_vehicles(vehicles: pd.DataFrame):
    # visualization

    # 1.
    # Bar Chart — Vehicle Category Distribution
    # Shows which vehicle types are most involved in accidents (cars vs motorcycles vs cycles vs trucks).

    plt.figure(figsize=(10,6))
    vehicles["vehicle_category_label"].value_counts().plot(kind="bar", color="royalblue", alpha=0.7)
    plt.title("Distribution of Vehicle Categories", fontsize=14)
    plt.xlabel("Vehicle Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    #Cars dominate, but motorcycles and cycles together form a large share.

    #2 
    # 2. Horizontal Bar Chart — Manoeuvre Distribution
    # Highlights the most common manoeuvres before accidents

    plt.figure(figsize=(8,5))
    vehicles["manoeuvre_label"].value_counts().plot(kind="barh", color="seagreen", alpha=0.7)
    plt.title("Distribution of Manoeuvres", fontsize=14)
    plt.xlabel("Count")
    plt.ylabel("Manoeuvre")
    plt.tight_layout()
    plt.show()

    # Straight‑ahead dominates, but turning and roundabouts are also major contributors

    #3. Bar Chart — Impact Point Distributio
    # Shows where vehicles are most often hit (front, side, rear)

    plt.figure(figsize=(10,6))
    vehicles["impact_point_label"].value_counts().plot(kind="bar", color="darkorange", alpha=0.7)
    plt.title("Distribution of Impact Points", fontsize=14)
    plt.xlabel("Impact Point")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Front collisions dominate, side impacts highlight intersection risks

    # 4. Stacked Bar Chart — Vehicle Category vs Impact Point
    # Combines two dimensions: which vehicle types tend to be hit where.

    impact_by_category = vehicles.groupby(["vehicle_category_label", "impact_point_label"]).size().unstack(fill_value=0)

    impact_by_category.plot(kind="bar", stacked=True, figsize=(12,7), colormap="tab20")
    plt.title("Impact Points by Vehicle Category", fontsize=14)
    plt.xlabel("Vehicle Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Impact Point", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    #Cars dominate impacts,but motorcycles have approximate 50% front impacts
    vehicles.columns


def visualize_user_vehicle(acc: pd.DataFrame):
    #### Phase 1 Visualizations (User/Vehicle-Level)
    fig, axes = plt.subplots(1, 2, figsize=(16,6))

    # 1. Injury Severity Distribution (fixed for seaborn v0.14+)
    sns.countplot(data=acc, 
                x="injury_severity_label", 
                hue="injury_severity_label",   # add hue
                palette="Set2", 
                ax=axes[0], 
                legend=False)                  # avoid duplicate legend
    axes[0].set_title("Distribution of Injury Severity (User-Level)")
    axes[0].tick_params(axis='x', rotation=45)

    # 2. Vehicle Category vs User Category (Heatmap)
    ct = pd.crosstab(acc["vehicle_category_label"], acc["user_category_label"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Vehicle Category vs User Category")

    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(16,6))

    # 1. Safety Equipment Usage vs Injury Severity (Counts)
    sns.countplot(data=acc, 
                x="safety_equipment_usage", 
                hue="injury_severity_label", 
                palette="Set1", ax=axes[0])
    axes[0].set_title("Safety Equipment Usage vs Injury Severity (Counts)")
    axes[0].tick_params(axis='x', rotation=45)

    # 2. Safety Equipment Usage vs Injury Severity (Normalized %)
    equip_severity = (
        acc.groupby(["safety_equipment_usage","injury_severity_label"])
        .size()
        .reset_index(name="count")
    )
    equip_severity["percentage"] = equip_severity.groupby("safety_equipment_usage")["count"].transform(lambda x: 100 * x / x.sum())

    sns.barplot(data=equip_severity, 
                x="safety_equipment_usage", 
                y="percentage", 
                hue="injury_severity_label", 
                palette="Set1", ax=axes[1])
    axes[1].set_title("Safety Equipment Usage vs Injury Severity (Normalized %)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(16,6))

    # 1. Seat Position vs Injury Severity (Counts)
    sns.countplot(data=acc, 
                x="seat_position_label", 
                hue="injury_severity_label", 
                palette="Paired", ax=axes[0])
    axes[0].set_title("Seat Position vs Injury Severity (Counts)")
    axes[0].tick_params(axis='x', rotation=45)

    # 2. Seat Position vs Injury Severity (Normalized %)
    seat_severity = (
        acc.groupby(["seat_position_label","injury_severity_label"])
        .size()
        .reset_index(name="count")
    )
    seat_severity["percentage"] = seat_severity.groupby("seat_position_label")["count"].transform(lambda x: 100 * x / x.sum())

    sns.barplot(data=seat_severity, 
                x="seat_position_label", 
                y="percentage", 
                hue="injury_severity_label", 
                palette="Paired", ax=axes[1])
    axes[1].set_title("Seat Position vs Injury Severity (Normalized %)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def visualize_accidents(acc: pd.DataFrame):
    #### Phase 2 Visualizations (Accident-Level Analysis)
    # de - duplication
    acc_unique = acc.drop_duplicates(subset="accident_id")
    print(acc.shape, acc_unique.shape)  # compare before/after
    print(acc_unique.info())

    ## 1 . Risk profiles by time and weather

    # --- Time of Day ---
    tod_raw = (
        acc_unique.groupby(["time_of_day","injury_severity_label"], observed=True)
                .size().reset_index(name="count")
    )
    tod_norm = tod_raw.copy()
    tod_norm["percentage"] = tod_norm.groupby("time_of_day", observed=True)["count"].transform(lambda x: 100*x/x.sum())
    tod_order = (
        tod_norm[tod_norm["injury_severity_label"]=="Killed"]
        .sort_values("percentage", ascending=False)["time_of_day"]
        .tolist()
    )

    fig, axes = plt.subplots(1,2, figsize=(16,6))
    sns.barplot(data=tod_raw, x="time_of_day", y="count", hue="injury_severity_label", ax=axes[0])
    axes[0].set_title("Time of Day × Severity (Raw Counts)")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=tod_norm, x="time_of_day", y="percentage", hue="injury_severity_label",
                order=tod_order, palette="coolwarm", ax=axes[1])
    axes[1].set_title("Time of Day × Severity (Normalized % / Sorted by Fatality)")
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Weather ---
    weather_raw = (
        acc_unique.groupby(["weather_label","injury_severity_label"], observed=True)
                .size().reset_index(name="count")
    )
    weather_norm = weather_raw.copy()
    weather_norm["percentage"] = weather_norm.groupby("weather_label", observed=True)["count"].transform(lambda x: 100*x/x.sum())
    weather_order = (
        weather_norm[weather_norm["injury_severity_label"]=="Killed"]
        .sort_values("percentage", ascending=False)["weather_label"]
        .tolist()
    )

    fig, axes = plt.subplots(1,2, figsize=(16,6))
    sns.barplot(data=weather_raw, x="weather_label", y="count", hue="injury_severity_label", ax=axes[0])
    axes[0].set_title("Weather × Severity (Raw Counts)")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=weather_norm, x="weather_label", y="percentage", hue="injury_severity_label",
                order=weather_order, palette="coolwarm", ax=axes[1])
    axes[1].set_title("Weather × Severity (Normalized % / Sorted by Fatality)")
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()



    #Temporal Trends 
    # --- 1. Facet bar plots: Accident counts per year, faceted by severity ---
    facet = sns.catplot(
        data=acc_unique, kind="count",
        col="injury_severity_label", x="year",
        col_wrap=2, height=4, aspect=1.2
    )

    facet.figure.suptitle("Accident Counts per Year by Injury Severity", y=1.05)

    # --- 2.  Collision Type × Injury Severity ---
    collision_ct = pd.crosstab(acc_unique["collision_label"], acc_unique["injury_severity_label"])
    collision_pct = collision_ct.div(collision_ct.sum(axis=1), axis=0) * 100

    # --- Plot side by side ---
    fig, axes = plt.subplots(1,2, figsize=(18,6))

    # Raw Count Heatmap
    sns.heatmap(collision_ct, annot=True, fmt="d", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Collision Type × Severity (Raw Counts)")
    axes[0].set_ylabel("Collision Type")
    axes[0].set_xlabel("Injury Severity")

    # Percentage Heatmap
    sns.heatmap(collision_pct, annot=True, fmt=".1f", cmap="YlOrBr", ax=axes[1])
    axes[1].set_title("Collision Type × Severity (Normalized %)")
    axes[1].set_ylabel("Collision Type")
    axes[1].set_xlabel("Injury Severity")

    plt.tight_layout()
    plt.show()


    # --- 3. Lighting ---
    lighting_raw = (
        acc_unique.groupby(["lighting_label","injury_severity_label"], observed=True)
                .size().reset_index(name="count")
    )
    lighting_norm = lighting_raw.copy()
    lighting_norm["percentage"] = lighting_norm.groupby("lighting_label", observed=True)["count"].transform(lambda x: 100*x/x.sum())
    lighting_order = (
        lighting_norm[lighting_norm["injury_severity_label"]=="Killed"]
        .sort_values("percentage", ascending=False)["lighting_label"]
        .tolist()
    )

    fig, axes = plt.subplots(1,2, figsize=(16,6))
    sns.barplot(data=lighting_raw, x="lighting_label", y="count", hue="injury_severity_label", ax=axes[0])
    axes[0].set_title("Lighting × Severity (Raw Counts)")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=lighting_norm, x="lighting_label", y="percentage", hue="injury_severity_label",
                order=lighting_order, palette="coolwarm", ax=axes[1])
    axes[1].set_title("Lighting × Severity (Normalized % / Sorted by Fatality)")
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    ### Road & Infrastructure

    # --- Road Layout ---
    layout_raw = (
        acc_unique.groupby(["road_layout_label","injury_severity_label"], observed=True)
                .size().reset_index(name="count")
    )
    layout_norm = layout_raw.copy()
    layout_norm["percentage"] = layout_norm.groupby("road_layout_label", observed=True)["count"].transform(lambda x: 100*x/x.sum())
    fatality_order = (
        layout_norm[layout_norm["injury_severity_label"]=="Killed"]
        .sort_values("percentage", ascending=False)["road_layout_label"]
        .tolist()
    )

    fig, axes = plt.subplots(1,2, figsize=(16,6))
    sns.barplot(data=layout_raw, x="road_layout_label", y="count", hue="injury_severity_label", ax=axes[0])
    axes[0].set_title("Road Layout × Severity (Raw Counts)")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=layout_norm, x="road_layout_label", y="percentage", hue="injury_severity_label",
                order=fatality_order, palette="coolwarm", ax=axes[1])
    axes[1].set_title("Road Layout × Severity (Normalized % / Sorted by Fatality)")
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Surface Condition ---
    surface_raw = (
        acc_unique.groupby(["surface_condition_label","injury_severity_label"], observed=True)
                .size().reset_index(name="count")
    )
    surface_norm = surface_raw.copy()
    surface_norm["percentage"] = surface_norm.groupby("surface_condition_label", observed=True)["count"].transform(lambda x: 100*x/x.sum())
    surface_order = (
        surface_norm[surface_norm["injury_severity_label"]=="Killed"]
        .sort_values("percentage", ascending=False)["surface_condition_label"]
        .tolist()
    )

    fig, axes = plt.subplots(1,2, figsize=(16,6))
    sns.barplot(data=surface_raw, x="surface_condition_label", y="count", hue="injury_severity_label", ax=axes[0])
    axes[0].set_title("Surface Condition × Severity (Raw Counts)")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=surface_norm, x="surface_condition_label", y="percentage", hue="injury_severity_label",
                order=surface_order, palette="coolwarm", ax=axes[1])
    axes[1].set_title("Surface Condition × Severity (Normalized % / Sorted by Fatality)")
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Intersection Heatmap ---
    intersection_ct = pd.crosstab(acc_unique["intersection_label"], acc_unique["injury_severity_label"])
    intersection_pct = intersection_ct.div(intersection_ct.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1,2, figsize=(18,6))
    sns.heatmap(intersection_ct, annot=True, fmt="d", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Intersection × Severity (Raw Counts)")

    sns.heatmap(intersection_pct, annot=True, fmt=".1f", cmap="YlOrBr", ax=axes[1])
    axes[1].set_title("Intersection × Severity (Normalized %)")
    plt.tight_layout()
    plt.show()
    ##Accident hour vs severity → distribution of accident times by severity

    fig, axes = plt.subplots(1,2, figsize=(16,6))

    # --- Boxplot ---
    sns.boxplot(data=acc_unique, 
                x="injury_severity_label", 
                y="hour", 
                hue="injury_severity_label", 
                palette="coolwarm", 
                legend=False, ax=axes[0])
    axes[0].set_title("Accident Hour Distribution by Injury Severity (Boxplot)")
    axes[0].set_xlabel("Injury Severity")
    axes[0].set_ylabel("Accident Hour (0–23)")
    axes[0].tick_params(axis='x', rotation=45)

    # --- Violin Plot ---
    sns.violinplot(data=acc_unique, 
                x="injury_severity_label", 
                y="hour", 
                hue="injury_severity_label", 
                palette="coolwarm", 
                legend=False, ax=axes[1])
    axes[1].set_title("Accident Hour Distribution by Injury Severity (Violin Plot)")
    axes[1].set_xlabel("Injury Severity")
    axes[1].set_ylabel("Accident Hour (0–23)")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    # fatal accident percentages by hour × lighting × weather

    acc_unique.loc[:, "light_weather"] = acc_unique["lighting_label"].astype(str) + " | " + acc_unique["weather_label"].astype(str)

    # Filter for fatal accidents
    fatal_only = acc_unique[acc_unique["injury_severity_label"] == "Killed"]

    # Group by hour and combined label
    compound_risk = (
        fatal_only.groupby(["hour", "light_weather"], observed=True)
                .size()
                .reset_index(name="count")
    )

    # Normalize within each hour
    compound_risk["percentage"] = compound_risk.groupby("hour", observed=True)["count"].transform(lambda x: 100 * x / x.sum())

    # Pivot for heatmap
    fatal_pivot = compound_risk.pivot(index="hour", columns="light_weather", values="percentage").fillna(0)

    # Plot
    plt.figure(figsize=(18,10))
    sns.heatmap(fatal_pivot, cmap="Reds", annot=True, fmt=".1f")
    plt.title("Fatal Accident Risk by Hour × Lighting × Weather (Normalized %)")
    plt.xlabel("Lighting × Weather")
    plt.ylabel("Hour of Day")
    plt.tight_layout()
    plt.show()