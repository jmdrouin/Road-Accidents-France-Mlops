import streamlit as st
import pandas as pd
import plotly.express as px
# plotly.graph_objects not used currently

# Page configuration
st.set_page_config(
    page_title="Road Accidents in France",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main title and subtitle
st.markdown('<p class="main-title">🚗 Road Accidents in France</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Data Science project on severity prediction</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Data Mining & Visualization", "Pre-processing & Feature engineering", "Modelling", "Conclusion"]
)

st.sidebar.markdown("---")
st.sidebar.info("📊 This project analyzes road accident data in France to predict accident severity.")

# Page 1: Data Mining & Visualization
if page == "Data Mining & Visualization":
    st.markdown('<p class="section-header">📊 Data Mining & Visualization</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Overview Section
    st.subheader("🔍 Data Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accidents", "839,985", help="Number of recorded accidents")
    with col2:
        st.metric("Persons involved", "1,876,005", help="Number of road users in dataset")
    with col3:
        st.metric("Vehicles involved", "1,433,389", help="Number of vehicles involved")
    with col4:
        st.metric("Features", "51", help="Number of features in dataset")
    with col5:
        st.metric("Time Period", "2005 to 2016", help="Data collection period")
    
    st.markdown("---")
    
    # Data Sample Section
    st.subheader("📋 Dataset Sample")
    
    # Load data with caching
    @st.cache_data
    def load_csv_samples():
        caracteristics = pd.read_csv('caracteristics.csv', encoding='latin-1', low_memory=False)
        # Force all columns to string to avoid mixed-type Arrow conversion issues (e.g., 'voie')
        places = pd.read_csv('places.csv', dtype=str, low_memory=False)
        users = pd.read_csv('users.csv', encoding='latin-1', low_memory=False)
        # Ensure encoding to avoid surprises on Windows locales
        vehicles = pd.read_csv('vehicles.csv', encoding='latin-1', low_memory=False)
        holidays = pd.read_csv('holidays.csv', encoding='latin-1', low_memory=False)
        return {
            'caracteristics': caracteristics.sample(5, random_state=42),
            'places': places.sample(5, random_state=42),
            'users': users.sample(5, random_state=42),
            'vehicles': vehicles.sample(5, random_state=42),
            'holidays': holidays.head(5)
        }
    
    try:
        samples = load_csv_samples()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Caracteristics", "📍 Places", "👥 Users", "🚗 Vehicles", "📅 Holidays"])
        
        with tab1:
            st.markdown("**Caracteristics Dataset** - General accident information")
            st.dataframe(samples['caracteristics'], use_container_width=True)
            st.caption(f"Showing 5 random samples from caracteristics.csv")
        
        with tab2:
            st.markdown("**Places Dataset** - Road and location details")
            st.dataframe(samples['places'], use_container_width=True)
            st.caption(f"Showing 5 random samples from places.csv")
        
        with tab3:
            st.markdown("**Users Dataset** - Information about people involved")
            st.dataframe(samples['users'], use_container_width=True)
            st.caption(f"Showing 5 random samples from users.csv")
        
        with tab4:
            st.markdown("**Vehicles Dataset** - Vehicle information")
            st.dataframe(samples['vehicles'], use_container_width=True)
            st.caption(f"Showing 5 random samples from vehicles.csv")
        
        with tab5:
            st.markdown("**Holidays Dataset** - French public holidays")
            st.dataframe(samples['holidays'], use_container_width=True)
            st.caption(f"Showing first 5 entries from holidays.csv")
        
        with st.expander("📚 Dataset Features Description"):
            st.write("""
            - **Temporal features**: Date, Time, Day of week and Holiday flag
            - **Location features**: GPS coordinates, Department, Municipality, urban/rural classification
            - **Road features**: Road type, Surface condition, Lighting conditions
            - **Weather features**: Weather conditions, Atmospheric conditions
            - **Accident features**: Collision type, Vehicles and persons involved, Safety equipment
            - **Severity**: Target variable (Unscathed, Light injury, Hospitalized, Fatal)
            """)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure CSV files are in the same directory as this script.")
    
    st.markdown("---")
    
    # Visualizations Section
    st.subheader("📈 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Temporal Analysis", "Geographical Analysis", "Feature Distributions", "Severity Analysis"])
    
    with tab1:
        st.markdown("### ⌚ Temporal Patterns")
        
        # Load temporal data with caching
        @st.cache_data
        def load_temporal_data():
            df = pd.read_csv('caracteristics.csv', encoding='latin-1', low_memory=False)
            # Convert key temporal columns to numeric safely
            for _col in ['an', 'hrmn', 'mois', 'jour']:
                if _col in df.columns:
                    df[_col] = pd.to_numeric(df[_col], errors='coerce')
                else:
                    df[_col] = pd.NA

            # Convert year from 2-digit to 4-digit (assuming 2000s) where appropriate
            def _fix_year(x):
                if pd.isna(x):
                    return x
                try:
                    x = int(x)
                except Exception:
                    return pd.NA
                return 2000 + x if x < 100 else x

            df['year'] = df['an'].apply(_fix_year)

            # Extract hour from hrmn (HHMM format) safely
            def _extract_hour(x):
                if pd.isna(x):
                    return pd.NA
                try:
                    x = int(x)
                except Exception:
                    return pd.NA
                if x < 0:
                    return pd.NA
                return int(x // 100)

            df['hour'] = df['hrmn'].apply(_extract_hour)

            # Map month numbers to month names
            month_map = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            df['month_name'] = df['mois'].map(month_map)

            # Create date column and extract day of week (coerce invalids)
            df['date'] = pd.to_datetime(
                df[['year', 'mois', 'jour']].rename({'mois': 'month', 'jour': 'day'}, axis=1),
                errors='coerce'
            )
            # Extract day name and weekday index without using .dt (for strict type-checkers)
            df['day_of_week'] = df['date'].apply(lambda d: d.strftime('%A') if pd.notna(d) else pd.NA)
            df['day_of_week_num'] = df['date'].apply(lambda d: d.weekday() if pd.notna(d) else pd.NA)
            return df
        
        try:
            df_temporal = load_temporal_data()
            
            # 1. Accidents by Year
            st.markdown("#### 📅 Accidents Over the Years")
            accidents_by_year = df_temporal.groupby('year').size().reset_index(name='count')
            fig_year = px.bar(
                accidents_by_year,
                x='year',
                y='count',
                title='Number of Accidents by Year',
                labels={'year': 'Year', 'count': 'Number of Accidents'},
                color='count',
                color_continuous_scale='Reds'
            )
            fig_year.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_year, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 2. Accidents by Month
                st.markdown("#### 📊 Accidents by Month")
                accidents_by_month = df_temporal.groupby(['mois', 'month_name']).size().reset_index(name='count')
                accidents_by_month = accidents_by_month.sort_values('mois')
                fig_month = px.bar(
                    accidents_by_month,
                    x='month_name',
                    y='count',
                    title='Number of Accidents by Month',
                    labels={'month_name': 'Month', 'count': 'Number of Accidents'},
                    color='count',
                    color_continuous_scale='Blues'
                )
                fig_month.update_layout(showlegend=False, height=400)
                fig_month.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_month, use_container_width=True)
            
            with col2:
                # 3. Accidents by Day of Week
                st.markdown("#### 📆 Accidents by Day of Week")
                df_weekday = df_temporal[df_temporal['day_of_week'].notna()].copy()
                accidents_by_weekday = df_weekday.groupby(['day_of_week_num', 'day_of_week']).size().reset_index(name='count')
                accidents_by_weekday = accidents_by_weekday.sort_values('day_of_week_num')
                fig_weekday = px.bar(
                    accidents_by_weekday,
                    x='day_of_week',
                    y='count',
                    title='Number of Accidents by Day of Week',
                    labels={'day_of_week': 'Day of Week', 'count': 'Number of Accidents'},
                    color='count',
                    color_continuous_scale='Oranges'
                )
                fig_weekday.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_weekday, use_container_width=True)
            
            # 4. Accidents by Time of Day
            st.markdown("#### 🕔 Accidents by Time of Day")
            df_temporal_clean = df_temporal[df_temporal['hour'].notna()].copy()
            accidents_by_hour = df_temporal_clean.groupby('hour').size().reset_index(name='count')
            accidents_by_hour = accidents_by_hour.sort_values('hour')
            fig_hour = px.area(
                accidents_by_hour,
                x='hour',
                y='count',
                title='Number of Accidents by Hour of Day',
                labels={'hour': 'Hour (24h format)', 'count': 'Number of Accidents'}
            )
            fig_hour.update_traces(fill='tozeroy', line_color='#2ca02c')
            fig_hour.update_layout(height=400)
            fig_hour.update_xaxes(dtick=2, range=[-0.5, 23.5])
            st.plotly_chart(fig_hour, use_container_width=True)
            
            # 5. Accidents on Holidays vs Regular Days
            st.markdown("#### 🎄 Accident Likelihood: Holidays vs Regular Weekdays")
            
            # Load and merge holiday data
            try:
                holidays_df = pd.read_csv('holidays.csv', encoding='latin-1')
                holidays_df['ds'] = pd.to_datetime(holidays_df['ds'], errors='coerce')
                holidays_df['is_holiday'] = True
                
                # Merge with temporal data
                df_with_holidays = df_temporal.merge(
                    holidays_df[['ds', 'is_holiday']], 
                    left_on='date', 
                    right_on='ds', 
                    how='left'
                )
                df_with_holidays['is_holiday'] = df_with_holidays['is_holiday'].fillna(False)
                
                # Filter to weekdays only (exclude weekends for fair comparison)
                df_weekdays_only = df_with_holidays[df_with_holidays['day_of_week_num'].isin([0, 1, 2, 3, 4])].copy()
                
                # Calculate statistics
                import numpy as np
                holiday_stats = df_weekdays_only.groupby('is_holiday').agg(
                    total_accidents=('date', 'count'),
                    unique_days=('date', 'nunique')
                ).reset_index()
                # Force recalculation with explicit Python division
                holiday_stats['avg_accidents_per_day'] = holiday_stats.apply(
                    lambda row: float(row['total_accidents']) / float(row['unique_days']), axis=1
                )
                holiday_stats['day_type'] = holiday_stats['is_holiday'].map({True: 'Holidays', False: 'Regular Weekdays'})
                
                # Create comparison chart
                fig_holiday = px.bar(
                    holiday_stats,
                    x='day_type',
                    y='avg_accidents_per_day',
                    title='Average Daily Accidents: Holidays vs Regular Weekdays',
                    labels={'day_type': 'Day Type', 'avg_accidents_per_day': 'Avg Accidents per Day'},
                    color='day_type',
                    color_discrete_map={'Holidays': '#e74c3c', 'Regular Weekdays': '#3498db'},
                    text='avg_accidents_per_day'
                )
                fig_holiday.update_traces(texttemplate='%{text:.2f}', textposition='outside', width=0.4)
                fig_holiday.update_layout(showlegend=False, height=450, margin=dict(t=80, b=40))
                st.plotly_chart(fig_holiday, use_container_width=True)
                
                # Show detailed statistics
                col1, col2 = st.columns(2)
                with col1:
                    regular_row = holiday_stats[holiday_stats['is_holiday'] == False].iloc[0]
                    regular_avg = float(regular_row['total_accidents']) / float(regular_row['unique_days'])
                    st.metric("Avg Accidents (Regular Weekdays)", f"{regular_avg:.0f}")
                with col2:
                    holiday_row = holiday_stats[holiday_stats['is_holiday'] == True].iloc[0]
                    holiday_avg = float(holiday_row['total_accidents']) / float(holiday_row['unique_days'])
                    pct_diff = ((holiday_avg - regular_avg) / regular_avg) * 100.0
                    st.metric(
                        "Avg Accidents (Holidays)", 
                        f"{holiday_avg:.0f}",
                        delta=f"{pct_diff:+.2f}%",
                        delta_color="inverse"
                    )
                
                st.caption("📊 Comparison limited to weekdays only (Mon-Fri) to ensure fair comparison")
                
            except Exception as e:
                st.warning(f"Could not load holiday data: {str(e)}")
            
            # Summary statistics
            with st.expander("📊 Temporal Statistics Summary"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Accidents", f"{len(df_temporal):,}")
                with col2:
                    if len(accidents_by_month) > 0:
                        peak_month = accidents_by_month.sort_values('count', ascending=False)['month_name'].iloc[0]
                        st.metric("Peak Month", str(peak_month))
                    else:
                        st.metric("Peak Month", "N/A")
                with col3:
                    if len(accidents_by_weekday) > 0:
                        peak_weekday = accidents_by_weekday.sort_values('count', ascending=False)['day_of_week'].iloc[0]
                        st.metric("Peak Weekday", str(peak_weekday))
                    else:
                        st.metric("Peak Weekday", "N/A")
                with col4:
                    if len(accidents_by_hour) > 0:
                        peak_hour_val = int(accidents_by_hour.sort_values('count', ascending=False)['hour'].iloc[0])
                        st.metric("Peak Hour", f"{peak_hour_val:02d}:00")
                    else:
                        st.metric("Peak Hour", "N/A")
                    
        except Exception as e:
            st.error(f"Error loading temporal data: {str(e)}")
            st.info("Please ensure caracteristics.csv is in the correct location.")
    
    with tab2:
        st.markdown("### 🌍 Geographical Distribution")
        
        # Load geographical data with caching
        @st.cache_data
        def load_geographical_data():
            df = pd.read_csv('caracteristics.csv', encoding='latin-1', low_memory=False)

            # Prepare numeric Lambert 93 coordinates
            df['lat_num'] = pd.to_numeric(df['lat'], errors='coerce')
            df['long_num'] = pd.to_numeric(df['long'], errors='coerce')
            # Department as numeric (coerce to handle unexpected strings)
            df['dep_num'] = pd.to_numeric(df['dep'], errors='coerce')

            # Keep rows with valid coords and department
            df_geo = df[
                (df['lat_num'].notna()) & (df['long_num'].notna()) &
                (df['lat_num'] != 0) & (df['long_num'] != 0) &
                df['dep_num'].notna()
            ].copy()

            # Convert EPSG:2154 (Lambert93) -> EPSG:4326 (WGS84)
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
                lon_wgs, lat_wgs = transformer.transform(df_geo['long_num'].values, df_geo['lat_num'].values)
                df_geo['longitude'] = lon_wgs
                df_geo['latitude'] = lat_wgs
            except Exception:
                # Fallback: assume scaled by 1e5
                df_geo['latitude'] = df_geo['lat_num'] / 100000
                df_geo['longitude'] = df_geo['long_num'] / 100000

            # Department code and mainland filter
            df_geo['dept_code'] = (df_geo['dep_num'] // 10).astype(int)
            mainland_depts = list(range(1, 20)) + list(range(21, 96))
            df_geo = df_geo[df_geo['dept_code'].isin(mainland_depts)].copy()

            # Mainland bounding box (final guardrail)
            df_geo = df_geo[(df_geo['latitude'] >= 41.0) & (df_geo['latitude'] <= 51.5) &
                            (df_geo['longitude'] >= -5.8) & (df_geo['longitude'] <= 9.8)].copy()

            # Robust per-department outlier clipping (removes rare off-sea points)
            def _clip_group(g: pd.DataFrame) -> pd.DataFrame:
                if len(g) < 50:
                    return g
                lat_q1, lat_q99 = g['latitude'].quantile([0.005, 0.995])
                lon_q1, lon_q99 = g['longitude'].quantile([0.005, 0.995])
                margin_lat, margin_lon = 0.1, 0.1
                return g[(g['latitude'] >= lat_q1 - margin_lat) & (g['latitude'] <= lat_q99 + margin_lat) &
                         (g['longitude'] >= lon_q1 - margin_lon) & (g['longitude'] <= lon_q99 + margin_lon)]

            df_geo = (
                df_geo.groupby('dept_code', group_keys=False)
                .apply(_clip_group)
                .reset_index(drop=True)
            )

            # Map dataset for plotting — ensure dept_code stays intact
            df_geo_map = df_geo.copy()
            if 'dept_code' not in df_geo_map.columns or df_geo_map['dept_code'].isna().any():
                df_geo_map['dept_code'] = (df_geo_map['dep_num'] // 10).astype(int)
            # 'agg' may not exist in all datasets; use safe access
            agg_series = df_geo_map.get('agg', pd.Series(index=df_geo_map.index))
            df_geo_map['area_type'] = agg_series.map({1: 'Rural', 2: 'Urban'}).fillna('Unknown')

            # Full dataset for stats (all accidents in mainland depts, BEFORE geographic filtering)
            df_all = df[df['dep_num'].notna()].copy()
            df_all['dept_code'] = (df_all['dep_num'] // 10).astype(int)
            df_all = df_all[df_all['dept_code'].isin(mainland_depts)].copy()
            agg_all = df_all.get('agg', pd.Series(index=df_all.index))
            df_all['area_type'] = agg_all.map({1: 'Rural', 2: 'Urban'}).fillna('Unknown')

            # Map department codes to names (simplified list for mainland France)
            dept_names = {
                1: 'Ain', 2: 'Aisne', 3: 'Allier', 4: 'Alpes-de-Haute-Provence', 5: 'Hautes-Alpes',
                6: 'Alpes-Maritimes', 7: 'Ardèche', 8: 'Ardennes', 9: 'Ariège', 10: 'Aube',
                11: 'Aude', 12: 'Aveyron', 13: 'Bouches-du-Rhône', 14: 'Calvados', 15: 'Cantal',
                16: 'Charente', 17: 'Charente-Maritime', 18: 'Cher', 19: 'Corrèze', 21: "Côte-d'Or",
                22: "Côtes-d'Armor", 23: 'Creuse', 24: 'Dordogne', 25: 'Doubs', 26: 'Drôme',
                27: 'Eure', 28: 'Eure-et-Loir', 29: 'Finistère', 30: 'Gard', 31: 'Haute-Garonne',
                32: 'Gers', 33: 'Gironde', 34: 'Hérault', 35: 'Ille-et-Vilaine', 36: 'Indre',
                37: 'Indre-et-Loire', 38: 'Isère', 39: 'Jura', 40: 'Landes', 41: 'Loir-et-Cher',
                42: 'Loire', 43: 'Haute-Loire', 44: 'Loire-Atlantique', 45: 'Loiret', 46: 'Lot',
                47: 'Lot-et-Garonne', 48: 'Lozère', 49: 'Maine-et-Loire', 50: 'Manche', 51: 'Marne',
                52: 'Haute-Marne', 53: 'Mayenne', 54: 'Meurthe-et-Moselle', 55: 'Meuse', 56: 'Morbihan',
                57: 'Moselle', 58: 'Nièvre', 59: 'Nord', 60: 'Oise', 61: 'Orne', 62: 'Pas-de-Calais',
                63: 'Puy-de-Dôme', 64: 'Pyrénées-Atlantiques', 65: 'Hautes-Pyrénées', 66: 'Pyrénées-Orientales',
                67: 'Bas-Rhin', 68: 'Haut-Rhin', 69: 'Rhône', 70: 'Haute-Saône', 71: 'Saône-et-Loire',
                72: 'Sarthe', 73: 'Savoie', 74: 'Haute-Savoie', 75: 'Paris', 76: 'Seine-Maritime',
                77: 'Seine-et-Marne', 78: 'Yvelines', 79: 'Deux-Sèvres', 80: 'Somme', 81: 'Tarn',
                82: 'Tarn-et-Garonne', 83: 'Var', 84: 'Vaucluse', 85: 'Vendée', 86: 'Vienne',
                87: 'Haute-Vienne', 88: 'Vosges', 89: 'Yonne', 90: 'Territoire de Belfort', 91: 'Essonne',
                92: 'Hauts-de-Seine', 93: 'Seine-Saint-Denis', 94: 'Val-de-Marne', 95: "Val-d'Oise"
            }
            df_geo_map['dept_name'] = df_geo_map['dept_code'].map(dept_names)
            df_all['dept_name'] = df_all['dept_code'].map(dept_names)
            
            return df_geo_map, df_all
        
        try:
            df_geo, df_all = load_geographical_data()
            
            st.info(f"📍 Showing {len(df_geo):,} accidents with valid coordinates on map (from {len(df_all):,} total mainland accidents)")
            
            # Sample data for better performance on map (show 10,000 points)
            sample_size = min(10000, len(df_geo))
            df_map = df_geo.sample(n=sample_size, random_state=42)
            
            # Sort so urban points (red) are drawn last and appear on top
            if 'area_type' in df_map.columns:
                df_map['area_type'] = pd.Categorical(
                    df_map['area_type'], categories=['Rural', 'Urban', 'Unknown'], ordered=True
                )
                df_map = df_map.sort_values('area_type', na_position='first')  # Rural first, Urban last
            
            # 1. Interactive Map with Urban/Rural distinction
            st.markdown("#### 🗺️ Accident Distribution Map (Sample of 10,000 accidents)")
            
            fig_map = px.scatter_mapbox(
                df_map,
                lat='latitude',
                lon='longitude',
                color='area_type',
                hover_data={'dept_name': True, 'dept_code': True, 'latitude': False, 'longitude': False},
                color_discrete_map={'Urban': '#e74c3c', 'Rural': '#3498db'},
                title='Accident Locations: Urban vs Rural',
                zoom=5,
                height=600
            )
            fig_map.update_traces(marker=dict(size=8))
            fig_map.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": 46.8, "lon": 2.5}
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 2. Top Departments by Accident Count (using ALL accidents)
                st.markdown("#### 📊 Top 20 Departments by Accident Count")
                dept_counts = df_all.groupby(['dept_code', 'dept_name']).size().reset_index(name='count')
                dept_counts = dept_counts.sort_values('count', ascending=False).head(20)
                
                fig_dept = px.bar(
                    dept_counts,
                    x='count',
                    y='dept_name',
                    orientation='h',
                    title='Accidents by Department (Top 20)',
                    labels={'dept_name': 'Department', 'count': 'Number of Accidents'},
                    color='count',
                    color_continuous_scale='Reds'
                )
                fig_dept.update_layout(showlegend=False, height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_dept, use_container_width=True)
            
            with col2:
                # 3. Urban vs Rural Distribution (using ALL accidents)
                st.markdown("#### 🏙️ Urban vs Rural Accident Distribution")
                urban_rural_counts = df_all.groupby('area_type').size().reset_index(name='count')
                
                fig_urban_rural = px.pie(
                    urban_rural_counts,
                    values='count',
                    names='area_type',
                    title='Proportion of Accidents by Area Type',
                    color='area_type',
                    color_discrete_map={'Urban': '#e74c3c', 'Rural': '#3498db'}
                )
                fig_urban_rural.update_traces(textposition='inside', textinfo='percent+label')
                fig_urban_rural.update_layout(height=500)
                st.plotly_chart(fig_urban_rural, use_container_width=True)
            
            # Summary statistics (using ALL accidents)
            with st.expander("📊 Geographical Statistics Summary"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Located Accidents", f"{len(df_all):,}", help="Total accidents in mainland France with valid department")
                with col2:
                    top_dept = dept_counts.iloc[0]
                    st.metric("Top Department", f"{top_dept['dept_name']}")
                with col3:
                    urban_count = (df_all['area_type'] == 'Urban').sum()
                    total_count = len(df_all)
                    urban_pct = (urban_count * 100.0) / total_count
                    st.metric("Urban Accidents", f"{urban_pct:.2f}%")
                with col4:
                    rural_count = (df_all['area_type'] == 'Rural').sum()
                    rural_pct = (rural_count * 100.0) / total_count
                    st.metric("Rural Accidents", f"{rural_pct:.2f}%")
                    
        except Exception as e:
            st.error(f"Error loading geographical data: {str(e)}")
            st.info("Please ensure caracteristics.csv is in the correct location.")
    
    with tab3:
        st.markdown("### 📊 Feature Distributions")
        
        # Load feature data with caching
        @st.cache_data
        def load_feature_data():
            # Load core accident-level datasets
            caracteristics = pd.read_csv('caracteristics.csv', encoding='latin-1', low_memory=False)
            places = pd.read_csv('places.csv', dtype=str, low_memory=False)
            users = pd.read_csv('users.csv', encoding='latin-1', low_memory=False)
            vehicles = pd.read_csv('vehicles.csv', encoding='latin-1', low_memory=False)

            # Keep the join key consistent everywhere
            for df_obj in (caracteristics, places, users, vehicles):
                df_obj['Num_Acc'] = df_obj['Num_Acc'].astype(str)

            # Accident-level view: caracteristics + places only (one row per accident)
            accidents = caracteristics.merge(places, on='Num_Acc', how='left')

            # Return all three granularities so charts can pull the right denominator
            return accidents, vehicles, users
        
        try:
            accidents_df, vehicles_df, users_df = load_feature_data()

            # Define label mappings for categorical features (accident-level)
            lighting_map = {
                1: 'Daylight',
                2: 'Twilight/Dawn',
                3: 'Night_without_public_lighting',
                4: 'Night_&_dysfunctional_lighting',
                5: 'Night_with_public_lighting_on'
            }
            weather_map = {
                1: 'Normal', 2: 'Light_rain', 3: 'Heavy_rain', 4: 'Snow/Hail',
                5: 'Fog/Smoke', 6: 'Strong_wind', 7: 'Dazzling', 8: 'Cloudy', 9: 'Other'
            }
            road_category_map = {
                1: 'Highway', 2: 'National_road', 3: 'Departmental_road',
                4: 'Communal_road', 5: 'Off_public_road', 6: 'Parking', 9: 'Other'
            }
            collision_map = {
                1: 'Frontal_two_vehicles',
                2: 'Rear_end_two_vehicles',
                3: 'Side_two_vehicles',
                4: 'Chain_three_plus',
                5: 'Multiple_three_plus',
                6: 'Other_collision',
                7: 'No_collision'
            }
            surface_condition_map = {
                1: 'Normal', 2: 'Wet', 3: 'Puddles', 4: 'Flooded',
                5: 'Snow', 6: 'Mud', 7: 'Ice', 8: 'Oil', 9: 'Other'
            }
            vehicle_category_map = {
                1: "Bicycle/Moped",
                2: "Bicycle/Moped",
                3: "other",
                4: "Motorcycle",
                5: "Motorcycle",
                6: "Motorcycle",
                7: "Car",
                8: "Utility(Truck/Trailer)",
                9: "Utility(Truck/Trailer)",
                10: "Utility(Truck/Trailer)",
                11: "Utility(Truck/Trailer)",
                12: "Utility(Truck/Trailer)",
                13: "Utility(Truck/Trailer)",
                14: "Utility(Truck/Trailer)",
                15: "Utility(Truck/Trailer)",
                16: "Tractor",
                17: "Tractor",
                18: "Bus/Coach",
                19: "Tram/Train",
                20: "other",
                21: "Tractor",
                30: "Motorcycle",
                31: "Motorcycle",
                32: "Motorcycle",
                33: "Motorcycle",
                34: "Motorcycle",
                35: "other",
                36: "other",
                37: "Bus/Coach",
                38: "Bus/Coach",
                39: "Tram/Train",
                40: "Tram/Train",
                99: "other",
            }
            # reduces 33 categories down to 8 groups (sorted for readability)

            # Apply mappings to accident-level dataframe
            accidents_df = accidents_df.copy()
            accidents_df['lighting_label'] = pd.to_numeric(accidents_df['lum'], errors='coerce').map(lighting_map).fillna('Unknown')
            accidents_df['weather_label'] = pd.to_numeric(accidents_df['atm'], errors='coerce').map(weather_map).fillna('Unknown')
            accidents_df['road_category_label'] = pd.to_numeric(accidents_df['catr'], errors='coerce').map(road_category_map).fillna('Unknown')
            accidents_df['collision_label'] = pd.to_numeric(accidents_df['col'], errors='coerce').map(collision_map).fillna('Unknown')
            accidents_df['surface_condition_label'] = pd.to_numeric(accidents_df['surf'], errors='coerce').map(surface_condition_map).fillna('Unknown')
            accidents_df['urban_label'] = pd.to_numeric(accidents_df['agg'], errors='coerce').map({1: 'Rural', 2: 'Urban'}).fillna('Unknown')

            # Vehicle-level mapping stays separate to preserve correct vehicle counts
            vehicles_df = vehicles_df.copy()
            vehicles_df['vehicle_category_label'] = pd.to_numeric(vehicles_df['catv'], errors='coerce').map(vehicle_category_map).fillna('Unknown')

            # Create two columns for side-by-side charts
            col1, col2 = st.columns(2)

            with col1:
                # 1. Lighting Conditions Distribution
                st.markdown("#### 💡 Lighting Conditions")
                lighting_counts = accidents_df['lighting_label'].value_counts().reset_index()
                lighting_counts.columns = ['Lighting', 'Count']

                # Ensure we order lighting categories by descending count so the largest bar appears first
                ordered_lighting = lighting_counts.sort_values('Count', ascending=False).head(8)['Lighting'].tolist()

                fig_lighting = px.bar(
                    lighting_counts.head(8),
                    x='Count',
                    y='Lighting',
                    orientation='h',
                    title='Distribution of Lighting Conditions',
                    labels={'Lighting': 'Lighting Condition', 'Count': 'Number of Accidents'},
                    color='Count',
                    color_continuous_scale='Viridis',
                    category_orders={'Lighting': ordered_lighting}
                )
                fig_lighting.update_layout(showlegend=False, height=350)
                fig_lighting.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_lighting, use_container_width=True)

                # 2. Weather Conditions Distribution
                st.markdown("#### 🌤️ Weather Conditions")
                weather_counts = accidents_df['weather_label'].value_counts().reset_index()
                weather_counts.columns = ['Weather', 'Count']

                fig_weather = px.bar(
                    weather_counts.head(8),
                    x='Weather',
                    y='Count',
                    title='Distribution of Weather Conditions',
                    labels={'Weather': 'Weather Condition', 'Count': 'Number of Accidents'},
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig_weather.update_layout(showlegend=False, height=350)
                fig_weather.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_weather, use_container_width=True)

                # 3. Surface Conditions Distribution
                st.markdown("#### ❄️ Road Surface Conditions")
                surface_counts = accidents_df['surface_condition_label'].value_counts().reset_index()
                surface_counts.columns = ['Surface', 'Count']

                fig_surface = px.bar(
                    surface_counts.head(8),
                    x='Surface',
                    y='Count',
                    title='Distribution of Road Surface Conditions',
                    labels={'Surface': 'Surface Condition', 'Count': 'Number of Accidents'},
                    color='Count',
                    color_continuous_scale='Oranges'
                )
                fig_surface.update_layout(showlegend=False, height=350)
                fig_surface.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_surface, use_container_width=True)

            with col2:
                # 4. Road Category Distribution
                st.markdown("#### 🛣️ Road Types")
                road_counts = accidents_df['road_category_label'].value_counts().reset_index()
                road_counts.columns = ['Road_Type', 'Count']

                fig_road = px.bar(
                    road_counts.head(7),
                    x='Count',
                    y='Road_Type',
                    orientation='h',
                    title='Distribution by Road Type',
                    labels={'Road_Type': 'Road Type', 'Count': 'Number of Accidents'},
                    color='Count',
                    color_continuous_scale='Greens'
                )
                fig_road.update_layout(showlegend=False, height=350, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_road, use_container_width=True)

                # 5. Collision Type Distribution
                st.markdown("#### 💥 Collision Types")
                collision_counts = accidents_df['collision_label'].value_counts().reset_index()
                collision_counts.columns = ['Collision_Type', 'Count']

                fig_collision = px.bar(
                    collision_counts.head(7),
                    x='Count',
                    y='Collision_Type',
                    orientation='h',
                    title='Distribution by Collision Type',
                    labels={'Collision_Type': 'Collision Type', 'Count': 'Number of Accidents'},
                    color='Count',
                    color_continuous_scale='Purples'
                )
                fig_collision.update_layout(showlegend=False, height=350, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_collision, use_container_width=True)

                # 6. Vehicle Category Distribution (Top 10, vehicle-level counts)
                st.markdown("#### 🚗 Vehicle Types (grouped)")
                vehicle_counts = vehicles_df['vehicle_category_label'].value_counts().reset_index()
                vehicle_counts.columns = ['Vehicle_Type', 'Count']

                fig_vehicle = px.bar(
                    vehicle_counts.head(10),
                    x='Count',
                    y='Vehicle_Type',
                    orientation='h',
                    title='Grouped Vehicle Types Involved',
                    labels={'Vehicle_Type': 'Vehicle Type', 'Count': 'Number of Vehicles'},
                    color='Count',
                    color_continuous_scale='Reds'
                )
                fig_vehicle.update_layout(showlegend=False, height=350, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_vehicle, use_container_width=True)

            # Summary statistics in an expander
            with st.expander("📊 Feature Distribution Summary"):
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    daylight_pct = (accidents_df['lighting_label'] == 'Daylight').sum() / len(accidents_df) * 100
                    st.metric("Daylight", f"{daylight_pct:.1f}%")
                with col2:
                    normal_weather_pct = (accidents_df['weather_label'] == 'Normal').sum() / len(accidents_df) * 100
                    st.metric("Normal Weather", f"{normal_weather_pct:.1f}%")
                with col3:
                    normal_surface_pct = (accidents_df['surface_condition_label'] == 'Normal').sum() / len(accidents_df) * 100
                    st.metric("Normal Surface", f"{normal_surface_pct:.1f}%")
                with col4:
                    # Percent of accidents involving at least one Car
                    accidents_with_car = vehicles_df[vehicles_df['vehicle_category_label'] == 'Car']['Num_Acc'].nunique()
                    car_pct = accidents_with_car / len(accidents_df) * 100 if len(accidents_df) > 0 else 0.0
                    st.metric("Car Accidents", f"{car_pct:.1f}%")
                with col5:
                    communal_pct = (accidents_df['road_category_label'] == 'Communal_road').sum() / len(accidents_df) * 100
                    st.metric("Communal Road", f"{communal_pct:.1f}%")
                with col6:
                    departmental_pct = (accidents_df['road_category_label'] == 'Departmental_road').sum() / len(accidents_df) * 100
                    st.metric("Departmental Road", f"{departmental_pct:.1f}%")
        
        except Exception as e:
            st.error(f"Error loading feature data: {str(e)}")
            st.info("Please ensure all CSV files are in the correct location.")
    
    with tab4:
        st.markdown("### 🚑 Severity Analysis")
        
        # Load severity data with caching
        @st.cache_data
        def load_severity_data():
            # Load datasets
            caracteristics = pd.read_csv('caracteristics.csv', encoding='latin-1', low_memory=False)
            places = pd.read_csv('places.csv', dtype=str, low_memory=False)
            users = pd.read_csv('users.csv', encoding='latin-1', low_memory=False)

            # Convert Num_Acc to string in all dataframes for consistent merging
            caracteristics['Num_Acc'] = caracteristics['Num_Acc'].astype(str)
            users['Num_Acc'] = users['Num_Acc'].astype(str)
            # places already loaded with dtype=str

            # Merge datasets using accident identifier only
            df = caracteristics.merge(places, on='Num_Acc', how='left')
            df = df.merge(users, on='Num_Acc', how='left')
            
            # Severity mapping (grav column in users.csv)
            severity_map = {
                1: 'Unscathed', 2: 'Fatal', 3: 'Hospitalized', 4: 'Light_injury'
            }
            df['severity'] = pd.to_numeric(df['grav'], errors='coerce').map(severity_map).fillna('Unknown')
            
            # Other mappings
            # Use the same lighting labels as in the Feature Distributions tab
            lighting_map = {
                1: 'Daylight',
                2: 'Twilight/Dawn',
                3: 'Night_without_public_lighting',
                4: 'Night_&_dysfunctional_lighting',
                5: 'Night_with_public_lighting_on'
            }
            weather_map = {
                1: 'Normal', 2: 'Light_rain', 3: 'Heavy_rain', 4: 'Snow/Hail',
                5: 'Fog/Smoke', 6: 'Strong_wind', 7: 'Dazzling', 8: 'Cloudy', 9: 'Other'
            }
            road_category_map = {
                1: 'Highway', 2: 'National_road', 3: 'Departmental_road',
                4: 'Communal_road', 5: 'Off_public_road', 6: 'Parking', 9: 'Other'
            }
            collision_map = {
                1: 'Frontal_two_vehicles',
                2: 'Rear_end_two_vehicles',
                3: 'Side_two_vehicles',
                4: 'Chain_three_plus',
                5: 'Multiple_three_plus',
                6: 'Other_collision',
                7: 'No_collision'
            }
            
            df['lighting_label'] = pd.to_numeric(df['lum'], errors='coerce').map(lighting_map).fillna('Unknown')
            df['weather_label'] = pd.to_numeric(df['atm'], errors='coerce').map(weather_map).fillna('Unknown')
            df['road_category_label'] = pd.to_numeric(df['catr'], errors='coerce').map(road_category_map).fillna('Unknown')
            df['collision_label'] = pd.to_numeric(df['col'], errors='coerce').map(collision_map).fillna('Unknown')
            df['urban_label'] = pd.to_numeric(df['agg'], errors='coerce').map({1: 'Rural', 2: 'Urban'}).fillna('Unknown')
            
            # Extract hour from hrmn
            df['hour'] = pd.to_numeric(df['hrmn'], errors='coerce') // 100
            df.loc[df['hour'] > 23, 'hour'] = pd.NA
            
            # Numeric encoding for correlation analysis
            df['severity_num'] = pd.to_numeric(df['grav'], errors='coerce')
            df['lighting_num'] = pd.to_numeric(df['lum'], errors='coerce')
            df['weather_num'] = pd.to_numeric(df['atm'], errors='coerce')
            df['road_category_num'] = pd.to_numeric(df['catr'], errors='coerce')
            df['collision_num'] = pd.to_numeric(df['col'], errors='coerce')
            df['urban_num'] = pd.to_numeric(df['agg'], errors='coerce')
            df['intersection_num'] = pd.to_numeric(df['int'], errors='coerce')
            # Add numeric encoding for surface condition to include in correlation analysis
            df['surface_condition_num'] = pd.to_numeric(df['surf'], errors='coerce')
            
            return df
        
        try:
            df_severity = load_severity_data()
            
            st.info(f"⚠️ Analyzing severity patterns across {len(df_severity):,} records")
            
            # Overall Severity Distribution
            st.markdown("#### 📊 Overall Severity Distribution")
            severity_counts = df_severity['severity'].value_counts().reset_index()
            severity_counts.columns = ['Severity', 'Count']
            
            # Reorder severity levels
            severity_order = ['Unscathed', 'Light_injury', 'Hospitalized', 'Fatal']
            severity_counts['Severity'] = pd.Categorical(severity_counts['Severity'], categories=severity_order, ordered=True)
            severity_counts = severity_counts.sort_values('Severity')
            
            fig_severity_dist = px.bar(
                severity_counts,
                x='Severity',
                y='Count',
                title='Distribution of Accident Severity',
                labels={'Severity': 'Severity Level', 'Count': 'Number of Cases'},
                color='Severity',
                color_discrete_map={
                    'Unscathed': '#2ecc71',
                    'Light_injury': '#f39c12',
                    'Hospitalized': '#e67e22',
                    'Fatal': '#e74c3c'
                }
            )
            fig_severity_dist.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_severity_dist, use_container_width=True)
            
            # Show severity percentages
            col1, col2, col3, col4 = st.columns(4)
            total = len(df_severity[df_severity['severity'] != 'Unknown'])
            with col1:
                unscathed_pct = (df_severity['severity'] == 'Unscathed').sum() / total * 100
                st.metric("Unscathed", f"{unscathed_pct:.1f}%", help="No injuries")
            with col2:
                light_pct = (df_severity['severity'] == 'Light_injury').sum() / total * 100
                st.metric("Light Injury", f"{light_pct:.1f}%", help="Minor injuries")
            with col3:
                hosp_pct = (df_severity['severity'] == 'Hospitalized').sum() / total * 100
                st.metric("Hospitalized", f"{hosp_pct:.1f}%", help="Serious injuries requiring hospitalization")
            with col4:
                fatal_pct = (df_severity['severity'] == 'Fatal').sum() / total * 100
                st.metric("Fatal", f"{fatal_pct:.1f}%", help="Fatal accidents", delta=None, delta_color="off")
            
            st.markdown("---")
            
            # Severity by various factors
            col1, col2 = st.columns(2)
            
            with col1:
                # Severity by Time of Day
                st.markdown("#### ⏰ Severity by Hour of Day")
                df_hour_severity = df_severity[df_severity['hour'].notna() & (df_severity['severity'] != 'Unknown')].copy()
                hour_severity = df_hour_severity.groupby(['hour', 'severity']).size().reset_index(name='count')
                hour_severity = hour_severity[hour_severity['severity'].isin(severity_order)]
                
                fig_hour_severity = px.bar(
                    hour_severity,
                    x='hour',
                    y='count',
                    color='severity',
                    title='Accident Severity by Hour',
                    labels={'hour': 'Hour of Day', 'count': 'Number of Cases', 'severity': 'Severity'},
                    color_discrete_map={
                        'Unscathed': '#2ecc71',
                        'Light_injury': '#f39c12',
                        'Hospitalized': '#e67e22',
                        'Fatal': '#e74c3c'
                    },
                    category_orders={'severity': severity_order}
                )
                fig_hour_severity.update_layout(height=400)
                st.plotly_chart(fig_hour_severity, use_container_width=True)
                
                # Severity by Road Type
                st.markdown("#### 🛣️ Severity by Road Type")
                df_road_severity = df_severity[df_severity['severity'] != 'Unknown'].copy()
                road_severity = df_road_severity.groupby(['road_category_label', 'severity']).size().reset_index(name='count')
                road_severity = road_severity[road_severity['severity'].isin(severity_order)]
                top_roads = road_severity.groupby('road_category_label')['count'].sum().nlargest(5).index
                road_severity = road_severity[road_severity['road_category_label'].isin(top_roads)]
                
                # Horizontal bar chart ordered by total counts (descending)
                ordered_roads = (
                    road_severity.groupby('road_category_label')['count']
                    .sum()
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )
                fig_road_severity = px.bar(
                    road_severity,
                    x='count',
                    y='road_category_label',
                    orientation='h',
                    color='severity',
                    title='Severity by Road Type (Top 5)',
                    labels={'road_category_label': 'Road Type', 'count': 'Number of Cases', 'severity': 'Severity'},
                    color_discrete_map={
                        'Unscathed': '#2ecc71',
                        'Light_injury': '#f39c12',
                        'Hospitalized': '#e67e22',
                        'Fatal': '#e74c3c'
                    },
                    category_orders={'road_category_label': ordered_roads, 'severity': severity_order}
                )
                fig_road_severity.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_road_severity, use_container_width=True)
            
            with col2:
                # Severity by Lighting Condition
                st.markdown("#### 💡 Severity by Lighting")
                lighting_severity = df_severity[df_severity['severity'] != 'Unknown'].groupby(['lighting_label', 'severity']).size().reset_index(name='count')
                lighting_severity = lighting_severity[lighting_severity['severity'].isin(severity_order)]
                top_lighting = lighting_severity.groupby('lighting_label')['count'].sum().nlargest(5).index
                lighting_severity = lighting_severity[lighting_severity['lighting_label'].isin(top_lighting)]
                
                # Horizontal lighting severity chart ordered by total counts (descending)
                ordered_lightings = (
                    lighting_severity.groupby('lighting_label')['count']
                    .sum()
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )
                fig_lighting_severity = px.bar(
                    lighting_severity,
                    x='count',
                    y='lighting_label',
                    orientation='h',
                    color='severity',
                    title='Severity by Lighting Condition',
                    labels={'lighting_label': 'Lighting', 'count': 'Number of Cases', 'severity': 'Severity'},
                    color_discrete_map={
                        'Unscathed': '#2ecc71',
                        'Light_injury': '#f39c12',
                        'Hospitalized': '#e67e22',
                        'Fatal': '#e74c3c'
                    },
                    category_orders={'lighting_label': ordered_lightings, 'severity': severity_order}
                )
                fig_lighting_severity.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_lighting_severity, use_container_width=True)
                
                # Severity by Urban/Rural
                st.markdown("#### 🏙️ Severity by Area Type")
                urban_severity = df_severity[df_severity['severity'] != 'Unknown'].groupby(['urban_label', 'severity']).size().reset_index(name='count')
                urban_severity = urban_severity[urban_severity['severity'].isin(severity_order)]
                urban_severity = urban_severity[urban_severity['urban_label'].isin(['Urban', 'Rural'])]
                
                # Order urban labels by total counts (descending)
                ordered_urban = (
                    urban_severity.groupby('urban_label')['count']
                    .sum()
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )
                fig_urban_severity = px.bar(
                    urban_severity,
                    x='count',
                    y='urban_label',
                    orientation='h',
                    color='severity',
                    title='Severity by Urban vs Rural',
                    labels={'urban_label': 'Area Type', 'count': 'Number of Cases', 'severity': 'Severity'},
                    color_discrete_map={
                        'Unscathed': '#2ecc71',
                        'Light_injury': '#f39c12',
                        'Hospitalized': '#e67e22',
                        'Fatal': '#e74c3c'
                    },
                    category_orders={'urban_label': ordered_urban, 'severity': severity_order}
                )
                fig_urban_severity.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_urban_severity, use_container_width=True)
            
            st.markdown("---")
            
            # Correlation Analysis
            st.markdown("#### 🔗 Feature Correlation with Severity")
            
            # Select numeric columns for correlation (use surface condition instead of intersection)
            corr_cols = ['severity_num', 'lighting_num', 'weather_num', 'road_category_num', 
                         'collision_num', 'urban_num', 'surface_condition_num', 'hour']
            df_corr = df_severity[corr_cols].dropna()
            
            if len(df_corr) > 0:
                # Calculate correlation matrix
                corr_matrix = df_corr.corr()
                
                # Create correlation heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(x="Feature", y="Feature", color="Correlation"),
                    x=['Severity', 'Lighting', 'Weather', 'Road_Type', 'Collision', 'Urban', 'Surface', 'Hour'],
                    y=['Severity', 'Lighting', 'Weather', 'Road_Type', 'Collision', 'Urban', 'Surface', 'Hour'],
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title='Correlation Matrix: Severity and Key Features',
                    zmin=-1,
                    zmax=1
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Show top correlations with severity
                severity_corr = corr_matrix['severity_num'].drop('severity_num').abs().sort_values(ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Correlations with Severity:**")
                    for feature, corr_value in severity_corr.head(5).items():
                        feature_names = {
                            'lighting_num': 'Lighting Condition',
                            'weather_num': 'Weather',
                            'road_category_num': 'Road Type',
                            'collision_num': 'Collision Type',
                            'urban_num': 'Urban/Rural',
                            'surface_condition_num': 'Surface Condition',
                            'hour': 'Hour of Day'
                        }
                        st.write(f"• **{feature_names.get(feature, feature)}**: {corr_matrix['severity_num'][feature]:.3f}")
                
                with col2:
                    st.markdown("**Key Insights:**")
                    st.write("• Positive correlation indicates higher values associated with more severe accidents")
                    st.write("• Negative correlation indicates inverse relationship")
                    st.write("• Values closer to ±1 indicate stronger relationships")
                    st.write("• Correlation does not imply causation")
            
            else:
                st.warning("Not enough data for correlation analysis")
        
        except Exception as e:
            st.error(f"Error loading severity data: {str(e)}")
            st.info("Please ensure all CSV files are in the correct location.")

# Page 2: Pre-processing & Feature Engineering
elif page == "Pre-processing & Feature engineering":
    st.markdown('<p class="section-header">🔧 Pre-processing & Feature Engineering</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Quality Section
    st.subheader("🧹 Data Quality Assessment")
    
    @st.cache_data
    def load_quality_data():
        df = pd.read_csv('caracteristics.csv', encoding='latin-1', low_memory=False)
        users = pd.read_csv('users.csv', encoding='latin-1', low_memory=False)
        vehicles = pd.read_csv('vehicles.csv', encoding='latin-1', low_memory=False)
        places = pd.read_csv('places.csv', dtype=str, low_memory=False)
        
        # Convert Num_Acc to string in all dataframes for consistent merging
        df['Num_Acc'] = df['Num_Acc'].astype(str)
        users['Num_Acc'] = users['Num_Acc'].astype(str)
        vehicles['Num_Acc'] = vehicles['Num_Acc'].astype(str)
        # places already loaded with dtype=str
        
        # Merge datasets using accident identifier only
        df_merged = df.merge(users, on='Num_Acc', how='left')
        df_merged = df_merged.merge(vehicles, on='Num_Acc', how='left')
        df_merged = df_merged.merge(places, on='Num_Acc', how='left')
        
        return df_merged
    
    try:
        df_quality = load_quality_data()
        
        # Overview metrics before preprocessing
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records (Raw)", f"{len(df_quality):,}", help="Initial number of records")
        with col2:
            st.metric("Features (Raw)", len(df_quality.columns), help="Initial number of features")
        with col3:
            missing_pct = (df_quality.isna().sum().sum() / (len(df_quality) * len(df_quality.columns))) * 100
            st.metric("Missing Data %", f"{missing_pct:.1f}%", help="Overall missing value percentage")
        with col4:
            duplicates = df_quality.duplicated().sum()
            st.metric("Duplicate Rows", f"{duplicates:,}", help="Number of duplicate records")
        
        st.markdown("---")
        
        # Missing Values Analysis
        st.markdown("#### 📊 Missing Values by Column (Top 20)")
        
        missing_data = (df_quality.isna().sum() / len(df_quality) * 100).sort_values(ascending=False).head(20)
        
        if len(missing_data) > 0:
            missing_df = pd.DataFrame({'Feature': missing_data.index, 'Missing %': missing_data.values})
            
            fig_missing = px.bar(
                missing_df,
                x='Missing %',
                y='Feature',
                orientation='h',
                title='Missing Values by Feature (Top 20)',
                labels={'Feature': 'Feature', 'Missing %': 'Percentage Missing'},
                color='Missing %',
                color_continuous_scale='Reds'
            )
            fig_missing.update_layout(showlegend=False, height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_missing, use_container_width=True)
        
        st.markdown("---")
        
        # Data Types Distribution
        st.markdown("#### 🔤 Data Types Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        dtype_counts = df_quality.dtypes.value_counts()
        
        with col1:
            numeric_cols = (df_quality.select_dtypes(include=['int64', 'int32', 'int16', 'int8', 'float64', 'float32']).shape[1])
            st.metric("Numeric Features", numeric_cols)
        
        with col2:
            categorical_cols = (df_quality.select_dtypes(include=['object']).shape[1])
            st.metric("Categorical Features", categorical_cols)
        
        with col3:
            datetime_cols = (df_quality.select_dtypes(include=['datetime64']).shape[1])
            st.metric("Datetime Features", datetime_cols)
        
        # Data type pie chart
        dtype_labels = []
        dtype_values = []
        for dtype, count in dtype_counts.items():
            dtype_labels.append(str(dtype))
            dtype_values.append(count)
        
        fig_dtypes = px.pie(
            values=dtype_values,
            names=dtype_labels,
            title='Distribution of Data Types',
            hole=0.4
        )
        fig_dtypes.update_traces(textposition='inside', textinfo='percent+label')
        fig_dtypes.update_layout(height=400)
        st.plotly_chart(fig_dtypes, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading quality data: {str(e)}")
    
    st.markdown("---")
    
    # Feature Engineering Section
    st.subheader("⚙️ Feature Engineering Pipeline")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Temporal", "Safety Equipment", "Vehicle & Impact", "Road Context"])
    
    with tab1:
        st.markdown("### ⏰ Temporal Feature Engineering")
        st.write("""
        **New Engineered Features:**
        - **date**: Combined from year, month, day columns
        - **day_of_week**: Extracted from date (Monday-Sunday)
        - **hour_group**: Binned hour into (Night, Morning, Afternoon, Evening)
        - **is_weekend**: Boolean flag for weekend days
        """)
        
        # Show example transformations
        with st.expander("📋 Transformation Examples"):
            example_data = {
                'Original Columns': ['year=2015, month=6, day=15, hour=1430', 'year=2015, month=6, day=20, hour=0800'],
                'date': ['2015-06-15', '2015-06-20'],
                'day_of_week': ['Monday', 'Saturday'],
                'hour_group': ['Afternoon', 'Morning'],
                'is_weekend': [False, True]
            }
            st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    with tab2:
        st.markdown("### 🛡️ Safety Equipment Feature Engineering")
        st.write("""
        **New Engineered Flags:**
        - **seatbelt_used**: True if Seat_belt type AND Used
        - **helmet_used**: True if Helmet type AND Used
        - **any_protection_used**: True if any safety equipment was used
        - **protection_effective**: True if protection used AND not undetermined
        """)
        
        with st.expander("📋 Transformation Examples"):
            example_data = {
                'safety_equipment_type': ['Seat_belt', 'Helmet', 'Not_used', 'Seat_belt'],
                'safety_equipment_usage': ['Used', 'Used', 'Not_used', 'Not_used'],
                'seatbelt_used': [True, False, False, False],
                'helmet_used': [False, True, False, False],
                'any_protection_used': [True, True, False, False],
                'protection_effective': [True, True, False, False]
            }
            st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    with tab3:
        st.markdown("### 🚗 Vehicle & Impact Feature Engineering")
        st.write("""
        **Vehicle Grouping:**
        - Car, Motorcycle, Bicycle, Truck, Bus, Other
        
        **Impact Grouping:**
        - Front (Front, Front_left, Front_right)
        - Rear (Rear, Rear_left, Rear_right)
        - Side (Left_side, Right_side)
        - Other
        
        **Interaction Features:**
        - **motorcycle_side_impact**: True if Motorcycle AND Side impact
        """)
        
        with st.expander("📋 Transformation Examples"):
            example_data = {
                'vehicle_category_label': ['Car', 'Motorcycle', 'Motorcycle', 'Truck'],
                'impact_point_label': ['Front', 'Side', 'Front', 'Rear'],
                'vehicle_group': ['Car', 'Motorcycle', 'Motorcycle', 'Truck'],
                'impact_group': ['Front', 'Side', 'Front', 'Rear'],
                'motorcycle_side_impact': [False, True, False, False]
            }
            st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    with tab4:
        st.markdown("### 🛣️ Road & Accident Context Feature Engineering")
        st.write("""
        **New Features Created:**
        - **is_night**: True for night conditions (with/without lighting)
        - **is_urban**: True for built-up areas
        - **lane_width**: Calculated as road_width / num_lanes
        - **road_group**: Simplified categories (Highway, Major_road, Local_road)
        - **weather_group**: Grouped weather (Clear, Rain, Snow, Fog, Other)
        """)
        
        with st.expander("📋 Transformation Examples"):
            example_data = {
                'lighting_label': ['Daylight', 'Night_with_lights', 'Night_no_lights', 'Dusk'],
                'urban_label': ['Built_up_area', 'Outside', 'Built_up_area', 'Outside'],
                'road_width': [14.0, 8.5, 10.2, 7.0],
                'num_lanes': [2, 1, 2, 1],
                'is_night': [False, True, True, False],
                'is_urban': [True, False, True, False],
                'lane_width': [7.0, 8.5, 5.1, 7.0]
            }
            st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    st.markdown("---")
    
    # Data Cleaning Section
    st.subheader("🧹 Data Cleaning & Column Pruning")
    
    with st.expander("📊 Columns Removed from Raw Dataset"):
        removed_cols = {
            'Category': [
                'Redundant Temporal', 'Redundant Temporal', 'Redundant Temporal', 'Redundant Location',
                'Redundant Location', 'Redundant Location', 'Sparse', 'Sparse', 'Sparse',
                'Sparse', 'Sparse', 'Sparse', 'Sparse', 'Sparse', 'Sparse', 'Sparse',
                'Skewed', 'Skewed', 'Skewed', 'Skewed'
            ],
            'Column': [
                'date', 'minute', 'time_hhmm', 'longitude_num', 'latitude_num', 'valid_geo',
                'mobile_obstacle_label', 'pedestrian_crossing_width', 'reserved_lane_label',
                'pedestrian_location_label', 'pedestrian_action_label', 'pedestrian_state_label',
                'birth_year', 'num_lanes', 'road_width',
                'infrastructure_label', 'school_zone_label', 'occupants_group',
                'situation_label', 'traffic_direction_label'
            ],
            'Reason': [
                'Replaced by derived features', 'Not used in analysis', 'Replaced by hour extraction',
                'Used for geospatial analysis only', 'Used for geospatial analysis only', 'Not needed',
                '~99.5% sparse', '~95% sparse', '~98% sparse',
                'Mostly "Not_pedestrian"', 'Mostly "Not_pedestrian"', 'Mostly "Not_pedestrian"',
                'Replaced by age + age_group', 'Captured in lane_width', 'Captured in lane_width',
                '~89% Unknown', '~39% Unknown', '~99% zeros',
                '~89% "On_road"', '~92% same direction'
            ]
        }
        st.dataframe(pd.DataFrame(removed_cols), use_container_width=True)
    
    st.markdown("---")
    
    # Dataset Transformation Summary
    st.subheader("📈 Dataset Transformation Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Raw Dataset")
        st.write("""
        - **Records**: 839,985
        - **Features**: 16 (raw columns)
        - **Missing Data**: ~15-20% average
        """)
    
    with col2:
        st.markdown("#### Processed Dataset")
        st.write("""
        - **Records**: 839,985
        - **Features**: ~35 (after engineering & pruning)
        - **Missing Data**: Reduced through imputation
        - **Quality**: Improved with derived features
        """)
    
    # Transformation steps visualization
    steps = ['Raw Data\n(16 features)', 'Feature\nEngineering\n(+20 features)', 
             'Remove\nRedundant\n(-10 features)', 'Final Dataset\n(~35 features)']
    feature_counts = [16, 36, 26, 35]
    
    fig_transform = px.bar(
        x=steps,
        y=feature_counts,
        title='Feature Count Through Processing Pipeline',
        labels={'x': 'Stage', 'y': 'Number of Features'},
        color=feature_counts,
        color_continuous_scale='Blues',
        text=feature_counts
    )
    fig_transform.update_traces(textposition='outside')
    fig_transform.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_transform, use_container_width=True)
    
    st.markdown("---")
    
    # Output Section
    st.subheader("💾 Final Output")
    
    st.info("""
    **Master Dataset Export**: `master_acc.csv`
    
    Contains all processed and engineered features ready for machine learning modeling.
    Includes:
    - ✅ All temporal engineered features
    - ✅ All safety equipment flags
    - ✅ Vehicle and impact groupings
    - ✅ Road context and weather features
    - ✅ Target variable: injury_severity_label
    """)

# Page 3: Modelling
elif page == "Modelling":
    st.markdown('<p class="section-header">🤖 Machine Learning Modelling</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Selection Section
    st.subheader("🎯 Model Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🚧 **Work in Progress**: Baseline models")
        st.write("Models to be tested:")
        st.write("- Logistic Regression")
        st.write("- Decision Tree")
        st.write("- Random Forest")
        st.write("- Naive Bayes")
    
    with col2:
        st.info("🚧 **Work in Progress**: Advanced models")
        st.write("Models to be tested:")
        st.write("- XGBoost")
        st.write("- LightGBM")
        st.write("- Neural Networks")
        st.write("- Ensemble methods")
    
    st.markdown("---")
    
    # Model Training Section
    st.subheader("🏋️ Model Training & Evaluation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Hyperparameter Tuning", "Feature Importance", "Predictions"])
    
    with tab1:
        st.markdown("### 📊 Performance Metrics")
        st.warning("🚧 **Coming Soon**: Comprehensive model evaluation metrics")
        st.write("Metrics to be displayed:")
        st.write("- **Classification metrics**: Accuracy, Precision, Recall, F1-Score")
        st.write("- **Confusion matrix**: Visual representation of predictions")
        st.write("- **ROC-AUC curves**: Multi-class classification performance")
        st.write("- **Cross-validation scores**: Model stability assessment")
        
        # Placeholder metrics
        st.markdown("#### Expected Performance Comparison")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Accuracy", "TBD", help="Highest accuracy achieved")
        with col2:
            st.metric("Best F1-Score", "TBD", help="Best F1 score across models")
        with col3:
            st.metric("Best Recall", "TBD", help="Best recall for severe accidents")
        with col4:
            st.metric("Best Precision", "TBD", help="Best precision for predictions")
    
    with tab2:
        st.markdown("### ⚙️ Hyperparameter Tuning")
        st.warning("🚧 **Coming Soon**: Hyperparameter optimization results")
        st.write("Tuning methods:")
        st.write("- Grid Search CV")
        st.write("- Random Search CV")
        st.write("- Bayesian Optimization")
        st.write("- Optuna framework")
        st.write("")
        st.write("Optimal parameters will be displayed for each model.")
    
    with tab3:
        st.markdown("### 🎯 Feature Importance")
        st.warning("🚧 **Coming Soon**: Analysis of most important features for predictions")
        st.write("Visualizations to include:")
        st.write("- Feature importance bar charts")
        st.write("- SHAP waterfall plots")
        st.write("- Partial dependence plots")
        st.write("- Feature interaction analysis")
    
    with tab4:
        st.markdown("### 🔮 Make Predictions")
        st.warning("🚧 **Coming Soon**: Interactive prediction interface")
        st.write("Features:")
        st.write("- Input accident parameters manually")
        st.write("- Get severity prediction from best model")
        st.write("- View prediction probabilities")
        st.write("- Explain prediction with SHAP values")
    
    st.markdown("---")
    
    # Model Comparison Section
    st.subheader("📈 Model Comparison")
    st.info("🚧 **Work in Progress**: Comprehensive comparison of all trained models")
    st.write("Will include:")
    st.write("- Performance metrics table comparing all models")
    st.write("- Training time comparison")
    st.write("- Overfitting analysis (train vs validation performance)")
    st.write("- Final model recommendation")

# Page 4: Conclusion
elif page == "Conclusion":
    st.markdown('<p class="section-header">📝 Conclusion</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Summary Section
    st.subheader("📊 Project Summary")
    st.info("🚧 **Work in Progress**: Overview of the complete analysis pipeline")
    st.write("""This section will summarize:""")
    st.write("- Dataset characteristics and preprocessing steps taken")
    st.write("- Feature engineering techniques applied")
    st.write("- Models tested and their performance comparison")
    st.write("- Best performing model and its metrics")
    
    st.markdown("---")
    
    # Key Findings Section
    st.subheader("🔍 Key Findings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.warning("🚧 **Coming Soon**: Data insights")
        st.write("Expected findings:")
        st.write("- Most critical factors affecting accident severity")
        st.write("- Temporal patterns and trends identified")
        st.write("- Geographical hotspots and risk zones")
        st.write("- Weather and road condition correlations")
    
    with col2:
        st.warning("🚧 **Coming Soon**: Model insights")
        st.write("Expected insights:")
        st.write("- Most predictive features for severity")
        st.write("- Model performance across severity classes")
        st.write("- Trade-offs between different models")
        st.write("- Model interpretability and explainability")
    
    st.markdown("---")
    
    # Recommendations Section
    st.subheader("💡 Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["Policy Recommendations", "Technical Improvements", "Deployment Strategy"])
    
    with tab1:
        st.markdown("### 🏛️ Policy & Safety Recommendations")
        st.info("🚧 **Work in Progress**: Actionable recommendations based on findings")
        st.write("Recommendations will include:")
        st.write("- Targeted interventions for high-risk locations")
        st.write("- Time-based safety measures (rush hour, night driving)")
        st.write("- Weather-related precautions and warnings")
        st.write("- Infrastructure improvements for accident-prone areas")
        st.write("- Public awareness campaigns based on data insights")
    
    with tab2:
        st.markdown("### 🔧 Model & Technical Improvements")
        st.info("🚧 **Work in Progress**: Areas for technical enhancement")
        st.write("Potential improvements:")
        st.write("- Incorporate additional data sources (traffic volume, road quality)")
        st.write("- Real-time prediction capabilities")
        st.write("- Deep learning architectures for spatial-temporal patterns")
        st.write("- Handling class imbalance more effectively")
        st.write("- Model ensemble and stacking strategies")
    
    with tab3:
        st.markdown("### 🚀 Deployment Strategy")
        st.info("🚧 **Work in Progress**: Production deployment considerations")
        st.write("Deployment plan:")
        st.write("- API development for model serving")
        st.write("- Integration with traffic management systems")
        st.write("- Real-time data pipeline architecture")
        st.write("- Model monitoring and retraining schedule")
        st.write("- Scalability and performance optimization")
    
    st.markdown("---")
    
    # Future Work Section
    st.subheader("🔮 Future Work")
    st.warning("🚧 **Coming Soon**: Roadmap for project extension")
    st.write("Future directions:")
    st.write("- **Extended temporal analysis**: Multi-year trend forecasting")
    st.write("- **Causal inference**: Understanding causality beyond correlation")
    st.write("- **Individual-level factors**: Driver behavior, vehicle age, etc.")
    st.write("- **Economic impact**: Cost analysis of accidents by severity")
    st.write("- **Comparative analysis**: Benchmarking with other European countries")
    st.write("- **Mobile application**: Real-time risk assessment for drivers")
    
    st.markdown("---")
    
    # Limitations Section
    st.subheader("⚠️ Limitations & Considerations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("📉 Data Limitations")
        st.write("Considerations:")
        st.write("- Reporting bias in accident data")
        st.write("- Missing values and data quality issues")
        st.write("- Temporal coverage and granularity")
        st.write("- Limited behavioral variables")
    
    with col2:
        st.info("🤖 Model Limitations")
        st.write("Considerations:")
        st.write("- Class imbalance in severity levels")
        st.write("- Generalization to unseen scenarios")
        st.write("- Model interpretability trade-offs")
        st.write("- Computational complexity constraints")
    
    st.markdown("---")
    
    # Final Remarks Section
    st.subheader("✅ Final Remarks")
    st.success("""This project demonstrates the application of machine learning techniques to predict road accident 
    severity in France. By analyzing historical accident data and building predictive models, we aim to provide 
    actionable insights for improving road safety and reducing accident severity. The interactive dashboard 
    enables stakeholders to explore the data, understand key patterns, and make data-driven decisions for 
    traffic safety interventions.""")
    
    st.write("")
    st.markdown("#### 📚 Technologies Used")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("- Python")
        st.write("- Pandas")
        st.write("- NumPy")
    with col2:
        st.write("- Scikit-learn")
        st.write("- XGBoost")
        st.write("- LightGBM")
    with col3:
        st.write("- Plotly")
        st.write("- Streamlit")
        st.write("- SHAP")
    with col4:
        st.write("- Jupyter")
        st.write("- Git")
        st.write("- VS Code")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚧 This application is under active development</p>
        <p>Road Accidents in France - Data Science Project 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)
