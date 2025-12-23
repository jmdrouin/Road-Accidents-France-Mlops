import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import pickle

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
    ["Data Mining & Visualization", "Pre-processing & Feature engineering", "Modeling & Optimization"]
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
                    top_dept_pct = (top_dept['count'] / len(df_all) * 100) if len(df_all) > 0 else 0.0
                    st.metric("Top Department", f"{top_dept['dept_name']} ({top_dept_pct:.1f}%)")
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

    @st.cache_data
    def load_pipeline_data():
        def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
            return df.drop(columns=[c for c in df.columns if c.startswith('Unnamed:')], errors='ignore')

        raw_caract = _drop_unnamed(pd.read_csv('caracteristics.csv', encoding='latin-1', low_memory=False))
        acc = _drop_unnamed(pd.read_csv('acc.csv', encoding='utf-8', low_memory=False))
        master = _drop_unnamed(pd.read_csv('master_acc.csv', encoding='utf-8', low_memory=False))

        return {
            'raw_caract': raw_caract,
            'acc': acc,
            'master': master,
        }

    try:
        data = load_pipeline_data()
        raw_caract = data['raw_caract']
        acc = data['acc']
        master = data['master']

        raw_rows, raw_cols = raw_caract.shape
        acc_rows, acc_cols = acc.shape
        master_rows, master_cols = master.shape
        unique_accidents = acc['accident_id'].nunique() if 'accident_id' in acc.columns else None

        removed_cols = [c for c in acc.columns if c not in master.columns]
        added_cols = [c for c in master.columns if c not in acc.columns]
        retained_cols = [c for c in master.columns if c in acc.columns]

        # =====================================================================
        # FEATURE TREATMENT BREAKDOWN & DATA FLOW FUNNEL
        # =====================================================================
        st.subheader("📊 Feature Treatment & Data Flow")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Funnel: raw merged data → cleaned → engineered → final
            stage_df = pd.DataFrame({
                'Stage': ['acc.csv\n(raw & merged)', 'acc.csv\n(cleaned)', 'acc.csv\n(engineered)', 'master_acc.csv\n(final)'],
                'Columns': [51, 35, 51, 27],
            })
            fig_funnel = px.funnel(stage_df, x="Columns", y="Stage", title="Feature flow: Merges → Cleaning → Engineering → Final")
            fig_funnel.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_funnel, use_container_width=True)
        
        with col_right:
            treatment_df = pd.DataFrame([
                {"Treatment": "🗑️ Removed", "Count": len(removed_cols)},
                {"Treatment": "✨ Engineered", "Count": len(added_cols)},
                {"Treatment": "✅ Retained", "Count": 11},
            ])
            fig_treat = px.pie(treatment_df, values="Count", names="Treatment",
                               title="Feature treatment breakdown",
                               color="Treatment",
                               color_discrete_map={"🗑️ Removed": "#e74c3c", "✨ Engineered": "#2ecc71", "✅ Retained": "#3498db"})
            fig_treat.update_traces(textposition='inside', textinfo='percent+label')
            fig_treat.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_treat, use_container_width=True)

        st.markdown("---")

        # =====================================================================
        # STEP-BY-STEP PIPELINE DOCUMENTATION
        # =====================================================================
        st.subheader("📋 Step-by-Step Pipeline")

        # Step 1
        with st.expander("**Step 1:** Data Loading & Initial Audit", expanded=False):
            st.markdown("""
            - Load `acc.csv` (merged accidents + places + vehicles + users + holidays)
            - Audit shape, dtypes, missing values, duplicates
            - Initial shape: **{:,} rows × {:,} columns**
            """.format(acc_rows, acc_cols))

        # Step 2
        with st.expander("**Step 2:** Drop Redundant & Sparse Columns", expanded=False):
            drop_step2 = [
                ("Unnamed: 0", "Index artifact"),
                ("vehicle_id", "Not needed for modeling"),
                ("date", "Will be rebuilt"),
                ("minute", "Unused after hour extraction"),
                ("time_hhmm", "Superseded by hour"),
                ("gps_label", "Redundant with coordinates"),
                ("holiday_name", "Binary flag sufficient"),
                ("longitude_num / latitude_num / valid_geo", "Spatial only"),
                ("mobile_obstacle_label", "~99% sparse"),
                ("pedestrian_crossing_width", "~95% sparse"),
                ("reserved_lane_label", "~98% sparse"),
                ("pedestrian_location/action/state_label", "Mostly 'Not_pedestrian'"),
            ]
            st.dataframe(pd.DataFrame(drop_step2, columns=["Column(s)", "Reason"]), use_container_width=True, hide_index=True)

        # Step 3
        with st.expander("**Step 3:** Type Casting", expanded=False):
            # Filter acc.csv to exclude columns dropped in Step 2 (representing cleaned state)
            dropped_cols_step2 = [
                'Unnamed: 0', 'vehicle_id', 'date', 'minute', 'time_hhmm', 'gps_label', 'holiday_name',
                'longitude_num', 'latitude_num', 'valid_geo', 'mobile_obstacle_label',
                'pedestrian_crossing_width', 'reserved_lane_label', 'pedestrian_location_label',
                'pedestrian_action_label', 'pedestrian_state_label'
            ]
            acc_cleaned = acc[[col for col in acc.columns if col not in dropped_cols_step2]]
            
            # Data types distribution (cleaned acc.csv)
            dtype_counts = acc_cleaned.dtypes.astype(str).value_counts().reset_index()
            dtype_counts.columns = ['Dtype', 'Count']
            max_count = int(dtype_counts['Count'].max()) if not dtype_counts.empty else 0
            fig_dtype = px.bar(
                dtype_counts, x='Dtype', y='Count',
                title='Data types in acc.csv (cleaned)',
                color='Dtype', text='Count'
            )
            fig_dtype.update_traces(textposition='outside')
            if max_count > 0:
                fig_dtype.update_yaxes(range=[0, max_count * 1.15])
            fig_dtype.update_layout(height=380, showlegend=False, margin=dict(t=50, b=30))
            fig_dtype.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_dtype, use_container_width=True)

        # Step 4
        with st.expander("**Step 4:** Numeric Cleaning & Age Derivation", expanded=False):
            st.markdown("""
            - **num_lanes**: values ≤0 or >10 → NaN
            - **road_width**: values ≤0 or >200 → NaN
            - **birth_year**: values <1900 or >2010 → NaN
            - **age** = year − birth_year
            - **age_group** = pd.cut(age, bins=[0,17,25,40,60,100], labels=[Child, Young_Adult, Adult, Middle_Aged, Senior])
            """)
            if 'age' in master.columns:
                fig_age = px.histogram(master[master['age'].notna()], x='age', nbins=40,
                                       title='Age distribution (after cleaning)',
                                       labels={'age': 'Age'}, color_discrete_sequence=['#3498db'])
                fig_age.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_age, use_container_width=True)

        # Step 5
        with st.expander("**Step 5:** Rare Category Grouping (<1% → 'other')", expanded=False):
            st.markdown("""
            Loop through all categorical columns; categories with <1% frequency are collapsed into **'other'**.
            This stabilizes model training by avoiding overfitting to rare levels.
            """)
            example_cols = ['vehicle_group', 'impact_group', 'road_group', 'weather_group']
            available = [c for c in example_cols if c in master.columns]
            if available:
                col1, col2 = st.columns(2)
                for i, c in enumerate(available[:4]):
                    vc = master[c].value_counts().reset_index()
                    vc.columns = [c, 'Count']
                    fig = px.bar(vc, x=c, y='Count', title=f'{c} distribution', text='Count',
                                 color_discrete_sequence=['#9b59b6'])
                    fig.update_traces(textposition='outside')
                    fig.update_layout(height=280, showlegend=False)
                    (col1 if i % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

        # Step 6
        with st.expander("**Step 6:** Feature Engineering", expanded=True):
            st.markdown("#### 🛡️ Safety Equipment Features")
            safety_eng = [
                ("seatbelt_used", "Seat_belt type AND Used"),
                ("helmet_used", "Helmet type AND Used"),
                ("any_protection_used", "Any equipment marked Used"),
                ("protection_effective", "Used AND not Undetermined"),
            ]
            st.dataframe(pd.DataFrame(safety_eng, columns=["Feature", "Logic"]), use_container_width=True, hide_index=True)
            # Visual logic: Safety equipment feature engineering
            safety_nodes = [
                'seatbelt_type', 'seatbelt_use', 'seatbelt_used',
                'helmet_type', 'helmet_use', 'helmet_used',
                'airbag_deployed', 'any_protection_used',
                'equipment_assessment', 'protection_effective'
            ]
            s_idx = {lbl: i for i, lbl in enumerate(safety_nodes)}
            safety_links = dict(
                source=[
                    s_idx['seatbelt_type'], s_idx['seatbelt_use'],
                    s_idx['helmet_type'], s_idx['helmet_use'],
                    s_idx['seatbelt_used'], s_idx['helmet_used'], s_idx['airbag_deployed'],
                    s_idx['any_protection_used'], s_idx['equipment_assessment']
                ],
                target=[
                    s_idx['seatbelt_used'], s_idx['seatbelt_used'],
                    s_idx['helmet_used'], s_idx['helmet_used'],
                    s_idx['any_protection_used'], s_idx['any_protection_used'], s_idx['any_protection_used'],
                    s_idx['protection_effective'], s_idx['protection_effective']
                ],
                value=[1]*9
            )
            fig_safety_logic = go.Figure(data=[go.Sankey(
                node=dict(label=safety_nodes),
                link=safety_links
            )])
            fig_safety_logic.update_layout(title='Safety equipment feature flow', height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig_safety_logic, use_container_width=True)

            st.markdown("#### 🚗 Vehicle & Impact Features")
            vehicle_eng = [
                ("vehicle_group", "Car / Motorcycle / Bicycle / Truck / Bus / other"),
                ("impact_group", "Front / Rear / Side / other"),
                ("motorcycle_side_impact", "Motorcycle AND Side impact"),
            ]
            st.dataframe(pd.DataFrame(vehicle_eng, columns=["Feature", "Logic"]), use_container_width=True, hide_index=True)
            # Weighted, decluttered Sankey with impact counts for Motorcycles
            if {'vehicle_group', 'impact_group'}.issubset(master.columns):
                moto = master[master['vehicle_group'] == 'Motorcycle']
                # Map impacts to main buckets to avoid clutter
                def _impact_bucket(val):
                    return val if val in ['Front', 'Rear', 'Side'] else 'other'
                impacts = moto['impact_group'].dropna().map(_impact_bucket)
                impact_counts = impacts.value_counts().reindex(['Front', 'Rear', 'Side', 'other'], fill_value=0)

                front, rear, side, other = [int(impact_counts.get(k, 0)) for k in ['Front', 'Rear', 'Side', 'other']]
                total = front + rear + side + other
                if total == 0:
                    st.info("No motorcycle records available to plot.")
                else:
                    nodes = [
                        'Motorcycle',
                        f'Impact: Front (n={front})',
                        f'Impact: Rear (n={rear})',
                        f'Impact: Side (n={side})',
                        f'Impact: Other (n={other})',
                        'motorcycle_side_impact = True',
                        'motorcycle_side_impact = False'
                    ]
                    idx = {lbl: i for i, lbl in enumerate(nodes)}
                    # Motorcycle → Impact buckets
                    sources_1 = [idx['Motorcycle']] * 4
                    targets_1 = [idx[f'Impact: Front (n={front})'], idx[f'Impact: Rear (n={rear})'],
                                 idx[f'Impact: Side (n={side})'], idx[f'Impact: Other (n={other})']]
                    values_1 = [front, rear, side, other]
                    colors_1 = ['#bdc3c7', '#bdc3c7', '#e74c3c', '#bdc3c7']
                    # Impact → Feature (Side → True, others → False)
                    sources_2 = [idx[f'Impact: Side (n={side})'], idx[f'Impact: Front (n={front})'],
                                 idx[f'Impact: Rear (n={rear})'], idx[f'Impact: Other (n={other})']]
                    targets_2 = [idx['motorcycle_side_impact = True'], idx['motorcycle_side_impact = False'],
                                 idx['motorcycle_side_impact = False'], idx['motorcycle_side_impact = False']]
                    values_2 = [side, front, rear, other]
                    colors_2 = ['#e74c3c', '#d0d3d4', '#d0d3d4', '#d0d3d4']

                    sankey = go.Figure(data=[go.Sankey(
                        node=dict(label=nodes),
                        link=dict(
                            source=sources_1 + sources_2,
                            target=targets_1 + targets_2,
                            value=values_1 + values_2,
                            color=colors_1 + colors_2
                        )
                    )])
                    sankey.update_layout(
                        title='Motorcycle-side impact: weighted flow with counts',
                        height=320, margin=dict(t=40, b=20, l=20, r=20)
                    )
                    st.plotly_chart(sankey, use_container_width=True)
            else:
                st.info("Columns `vehicle_group` and `impact_group` are required for the weighted flow.")

            st.markdown("#### 🛣️ Road & Context Features")
            road_eng = [
                ("is_night", "Lighting in [Night_with_*, Night_without_*]"),
                ("is_urban", "urban_label == Built_up_area"),
                ("lane_width", "road_width / num_lanes"),
                ("road_group", "Highway / Major_road / Local_road / other"),
                ("weather_group", "Clear / Rain / Snow / Fog / other"),
            ]
            st.dataframe(pd.DataFrame(road_eng, columns=["Feature", "Logic"]), use_container_width=True, hide_index=True)
            # Road & Context visuals removed to reduce clutter; logic documented in the table above.

            st.markdown("#### ⏰ Temporal Features")
            temporal_eng = [
                ("date", "Rebuilt from year/month/day"),
                ("day_of_week", "Monday–Sunday from date"),
                ("hour_group", "Night [0-6) / Morning [6-12) / Afternoon [12-18) / Evening [18-24)"),
                ("is_weekend", "Saturday or Sunday"),
            ]
            st.dataframe(pd.DataFrame(temporal_eng, columns=["Feature", "Logic"]), use_container_width=True, hide_index=True)
            # Visual logic: Temporal feature engineering
            t_labels = [
                'year', 'month', 'day', 'date', 'day_of_week', 'hour', 'Night', 'Morning', 'Afternoon', 'Evening', 'Unknown', 'is_weekend'
            ]
            t_idx = {lbl: i for i, lbl in enumerate(t_labels)}
            t_links = dict(
                source=[
                    t_idx['year'], t_idx['month'], t_idx['day'],
                    t_idx['date'], t_idx['hour'], t_idx['hour'], t_idx['hour'], t_idx['hour'], t_idx['hour'],
                    t_idx['day_of_week']
                ],
                target=[
                    t_idx['date'], t_idx['date'], t_idx['date'],
                    t_idx['day_of_week'], t_idx['Night'], t_idx['Morning'], t_idx['Afternoon'], t_idx['Evening'], t_idx['Unknown'],
                    t_idx['is_weekend']
                ],
                value=[1]*10
            )
            fig_t_logic = go.Figure(data=[go.Sankey(node=dict(label=t_labels), link=t_links)])
            fig_t_logic.update_layout(title='Temporal feature flow', height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig_t_logic, use_container_width=True)

        # Step 7
        with st.expander("**Step 7:** Final Pruning (drop superseded columns)", expanded=False):
            pruned = [
                ("safety_equipment_type / usage", "Replaced by seatbelt_used, helmet_used, etc."),
                ("vehicle_category_label", "Replaced by vehicle_group"),
                ("impact_point_label", "Replaced by impact_group"),
                ("lighting_label / urban_label", "Replaced by is_night, is_urban"),
                ("road_category_label / weather_label", "Replaced by road_group, weather_group"),
                ("year / month / day / hour / time_of_day", "Replaced by date, day_of_week, hour_group"),
                ("birth_year / num_lanes / road_width", "Replaced by age, lane_width"),
                ("infrastructure_label (~89% Unknown)", "Low signal"),
                ("school_zone_label (~39% Unknown)", "Low signal"),
                ("occupants_group (~99% zeros)", "Low signal"),
                ("situation_label (~89% On_road)", "Skewed"),
                ("traffic_direction_label (~92% Same)", "Skewed"),
                ("intersection/traffic_regime/road_profile/road_layout", "Skewed"),
            ]
            st.dataframe(pd.DataFrame(pruned, columns=["Column(s)", "Reason"]), use_container_width=True, hide_index=True)

        st.markdown("---")

        # =====================================================================
        # FINAL FEATURE DISTRIBUTIONS
        # =====================================================================
        st.subheader("📈 Final Feature Distributions (master_acc.csv)")

        # Categorical distributions
        st.markdown("#### Categorical Features")
        cat_cols_final = [c for c in master.columns if master[c].dtype.name in ['object', 'category'] or master[c].nunique() < 15]
        # Prioritize key engineered features
        priority_cats = ['injury_severity_label', 'vehicle_group', 'impact_group', 'weather_group',
                         'road_group', 'age_group', 'hour_group', 'day_of_week', 'collision_label',
                         'surface_condition_label', 'manoeuvre_label', 'sex_label', 'user_category_label']
        display_cats = [c for c in priority_cats if c in master.columns][:8]

        if display_cats:
            rows = (len(display_cats) + 1) // 2
            for row_idx in range(rows):
                col1, col2 = st.columns(2)
                for col_idx, col_container in enumerate([col1, col2]):
                    feat_idx = row_idx * 2 + col_idx
                    if feat_idx < len(display_cats):
                        feat = display_cats[feat_idx]
                        vc = master[feat].value_counts().head(10).reset_index()
                        vc.columns = [feat, 'Count']
                        fig = px.bar(vc, y=feat, x='Count', orientation='h',
                                     title=f'{feat}', text='Count',
                                     color_discrete_sequence=['#1abc9c'])
                        fig.update_traces(textposition='outside')
                        fig.update_layout(height=280, showlegend=False, yaxis={'categoryorder': 'total ascending'})
                        col_container.plotly_chart(fig, use_container_width=True)

        # Boolean features
        st.markdown("#### Boolean Features")
        bool_cols = [c for c in master.columns if master[c].dtype == bool]
        if bool_cols:
            bool_counts = []
            for bc in bool_cols:
                true_pct = master[bc].mean() * 100
                bool_counts.append({'Feature': bc, 'True %': true_pct, 'False %': 100 - true_pct})
            bool_df = pd.DataFrame(bool_counts)
            max_val = float(bool_df['True %'].max()) if len(bool_df) > 0 else 0.0
            fig_bool = px.bar(bool_df, x='Feature', y='True %', title='Boolean feature True rates (%)',
                              text='True %', color_discrete_sequence=['#e67e22'])
            fig_bool.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            if max_val > 0:
                fig_bool.update_yaxes(range=[0, max(1.0, max_val * 1.15)])
            fig_bool.update_layout(height=420, showlegend=False, margin=dict(t=60, b=40))
            fig_bool.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_bool, use_container_width=True)

        # Numeric features
        st.markdown("#### Numeric Features")
        num_cols_final = master.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float32', 'float64']).columns.tolist()
        priority_nums = ['age', 'lane_width', 'season']
        display_nums = [c for c in priority_nums if c in num_cols_final][:3]

        if display_nums:
            cols = st.columns(len(display_nums))
            for i, feat in enumerate(display_nums):
                with cols[i]:
                    fig = px.histogram(master[master[feat].notna()], x=feat, nbins=30,
                                       title=f'{feat} distribution',
                                       color_discrete_sequence=['#8e44ad'])
                    fig.update_layout(height=280, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Final summary
        st.subheader("✅ Ready for Modeling")
        st.info(f"""
        **master_acc.csv** contains **{master_rows:,} rows** × **27 columns** ready for ML.
        
        Key engineered features: vehicle_group, impact_group, weather_group, road_group, age_group, hour_group, 
        is_night, is_urban, seatbelt_used, helmet_used, motorcycle_side_impact, lane_width.
        
        Target variable: **injury_severity_label**
        """)

    except Exception as e:
        st.error(f"Error loading preprocessing data: {str(e)}")
        st.info("Please ensure acc.csv and master_acc.csv exist alongside the app.")

# Page 3: Modeling & Optimization
elif page == "Modeling & Optimization":
    st.markdown('<p class="section-header">🚀 Modeling & Optimization</p>', unsafe_allow_html=True)
    st.markdown("### From Baseline to Production-Ready Models")
    
    st.markdown("---")
    
    # Overview Metrics
    st.subheader("📊 Project Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", "1,412,032")
    with col2:
        st.metric("Features", "85")
    with col3:
        st.metric("Target Classes", "4 / 2")
    with col4:
        st.metric("Best ROC-AUC", "83.4%")
    
    st.markdown("---")
    
    # Section 1: Problem Definition
    with st.expander("🎯 1. Problem Definition & Approach", expanded=True):
        st.markdown("""
        #### The Challenge
        Predict **injury severity** in road accidents using historical French accident data (2005-2016).
        
        **Target Variable:** `injury_severity_label`
        - 0️⃣ Hospitalized
        - 1️⃣ Slight Injury  
        - 2️⃣ Killed
        - 3️⃣ Uninjured
        
        #### Why This Matters
        - 🚑 **Emergency Response**: Prioritize severe cases
        - 📊 **Policy Making**: Identify high-risk patterns
        - 🛡️ **Prevention**: Target safety interventions
        
        #### Key Challenges
        - ⚖️ **Severe Class Imbalance**: Killed class represents <3% of data
        - 🔀 **Multi-class complexity**: 4 distinct severity levels
        - 📏 **High dimensionality**: 85 features after encoding
        - 🎭 **Mixed data types**: Categorical, numerical, and binary features
        """)
    
    st.markdown("---")
    
    # Section 2: Data Preparation Pipeline
    with st.expander("🔧 2. Data Preparation Pipeline"):
        st.markdown("""
        #### Train/Test Split
        - **Train**: 1,129,625 samples (80%)
        - **Test**: 282,407 samples (20%)
        - **Strategy**: Stratified to preserve class distribution
        
        #### Preprocessing Steps
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Numeric Features (2)**
            - `age`, `lane_width`
            - **Imputation**: Median strategy
            - **Scaling**: StandardScaler
            
            **Categorical Features (15)**
            - collision_label, surface_condition_label, manoeuvre_label, etc.
            - **Imputation**: Most frequent category
            - **Encoding**: OneHotEncoder → 74 features
            - **Special handling**: `age_group` (Unknown category for NaNs)
            """)
        
        with col2:
            st.markdown("""
            **Binary Features (9)**
            - is_weekend, is_holiday, seatbelt_used, helmet_used
            - any_protection_used, protection_effective
            - motorcycle_side_impact, is_night, is_urban
            - **No transformation needed**
            
            **Final Feature Matrix**
            - 2 (numeric) + 74 (encoded) + 9 (binary) = **85 features**
            - All numeric, ready for ML
            """)
        
        st.info("✅ **Result**: Zero missing values, fully numeric dataset ready for modeling")
    
    st.markdown("---")
    
    # Section 3: Model Selection Journey
    with st.expander("🔍 3. Model Selection Journey"):
        st.markdown("""
        #### 3.1 LazyPredict Screening
        Quick benchmarking of 15+ classifiers on imbalanced data to identify promising candidates.
        
        **Top 5 Models**:
        """)
        
        # LazyPredict results table
        lazy_data = {
            "Model": ["LGBMClassifier", "LogisticRegression", "CalibratedClassifierCV", 
                     "LinearDiscriminantAnalysis", "RidgeClassifier"],
            "Accuracy": [0.65, 0.64, 0.64, 0.63, 0.63],
            "Time (s)": [33.05, 17.33, 288.20, 11.90, 12.85],
            "Notes": ["🥇 Best overall", "Fast baseline", "Excellent calibration", 
                     "Lightweight", "Robust linear"]
        }
        st.table(pd.DataFrame(lazy_data))
        
        st.markdown("""
        #### 3.2 Deep Evaluation (200K Sample, 3-Fold CV)
        Selected 5 models evaluated with stratified cross-validation:
        """)
        
        # Detailed benchmarking results
        bench_data = {
            "Model": ["LightGBM", "LinearDiscriminantAnalysis", "LogisticRegression", 
                     "CalibratedClassifierCV", "RidgeClassifier"],
            "Accuracy": [0.6507, 0.6348, 0.5672, 0.6269, 0.6307],
            "Macro F1": [0.4683, 0.4696, 0.4450, 0.4429, 0.4094],
            "Balanced Accuracy": [0.4608, 0.4583, 0.5377, 0.4393, 0.4134]
        }
        st.dataframe(pd.DataFrame(bench_data).style.highlight_max(subset=["Accuracy", "Macro F1", "Balanced Accuracy"], axis=0, color='#DAA520'))
        
        st.success("🏆 **Winner**: LightGBM - Best accuracy and F1, handles non-linear patterns excellently")
    
    st.markdown("---")
    
    # Section 4: Multiclass Modeling
    with st.expander("📊 4. Multiclass Classification (4 Severity Levels)", expanded=True):
        st.markdown("### 4.1 Baseline LightGBM (No Resampling)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Configuration:**
            - 500 trees, learning_rate=0.05
            - is_unbalance=True
            - 5-fold Stratified CV
            
            **Performance:**
            - ✅ Accuracy: 0.6535
            - ⚠️ Macro F1: 0.4677
            - ⚠️ Balanced Accuracy: 0.4620
            """)
        
        with col2:
            st.markdown("""
            **Per-Class F1 Scores:**
            - Hospitalized (0): 0.46
            - Slight Injury (1): 0.56
            - 🔴 **Killed (2): 0.05** ⚠️
            - Uninjured (3): 0.79
            
            **Problem**: Severe underfitting of minority class (Killed)
            """)
        
        st.markdown("---")
        st.markdown("### 4.2 SMOTE Oversampling")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Approach**: Fully balance all 4 classes
            
            **Class Distribution**:
            - Before: [215K, 363K, 29K, 523K]
            - After: [523K, 523K, 523K, 523K]
            - Training samples: 1.13M → 2.09M
            """)
        
        with col2:
            st.markdown("""
            **Performance**:
            - Accuracy: 0.64
            - Macro F1: 0.50 ↑
            - Balanced Accuracy: 0.49 ↑
            
            **Killed class (2)**: F1 = 0.20 (↑ from 0.05)
            """)
        
        st.warning("⚠️ Full balancing introduced noise, slightly reduced precision for majority classes")
        
        st.markdown("---")
        st.markdown("### 4.3 Borderline SMOTE (Targeted Oversampling)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Custom Sampling Strategy**:
            - Class 0: 350,000
            - Class 1: 363,089 (unchanged)
            - Class 2: 300,000 
            - Class 3: 522,866
            
            **Rationale**: Focus on decision boundaries, avoid over-balancing
            """)
        
        with col2:
            st.markdown("""
            **Performance**:
            - Accuracy: 64.3%
            - Macro F1: 50.1%
            - Balanced Accuracy: 49.5%
            
            **Per-Class F1 Scores**:
            - Hospitalized (0): 0.44
            - Slight Injury (1): 0.55
            - Killed (2): 0.23 (↑ from 0.05)
            - Uninjured (3): 0.78
            """)
        
        st.success("✅ **Best Multiclass Model**: Borderline SMOTE with custom sampling achieves best balance")
        
        st.markdown("---")
        st.markdown("### 4.4 Advanced Optimization Attempts")
        
        st.markdown("""
        **Class Weighting**:
        - Applied custom weights: {0: 1.5, 1: 1.0, 2: 3.0, 3: 0.8}
        - Result: Similar to Borderline SMOTE, no significant improvement
        
        **Threshold Tuning**:
        - Per-class threshold optimization using validation F1
        - Result: Marginal gains, increased complexity
        
        **Feature Selection (SHAP-based)**:
        - Reduced to top 40 features
        - Result: Faster inference but lower recall for minority classes
        
        **Hyperparameter Search**:
        - RandomizedSearchCV (15 iterations, 3-fold CV)
        - Best params: num_leaves=127, n_estimators=800, learning_rate=0.1
        - Result: Minimal improvement over Borderline SMOTE baseline
        """)
        
        st.info("💡 **Key Insight**: For multiclass with severe imbalance, targeted oversampling (Borderline SMOTE) outperforms complex optimization")
    
    st.markdown("---")
    
    # Section 5: Binary Classification
    with st.expander("⚖️ 5. Binary Classification (Severe vs Not Severe)", expanded=True):
        st.markdown("""
        ### Problem Reformulation
        Collapse 4 classes into 2:
        - **Severe (1)**: Hospitalized (0) + Killed (2)
        - **Not Severe (0)**: Slight Injury (1) + Uninjured (3)
        
        **Motivation**: Simplify for deployment, improve recall for critical cases
        """)
        
        st.markdown("---")
        st.markdown("### 5.1 Binary Model Comparison")
        
        # Binary results table
        binary_data = {
            "Model": ["Baseline LGBM", "LGBM + Class Weights", "LGBM + SMOTE", "LGBM + Borderline SMOTE"],
            "Accuracy": [0.8241, 0.7394, 0.8161, 0.8137],
            "F1 (Severe)": [0.4852, 0.5657, 0.5192, 0.5169],
            "Balanced Acc": [0.6647, 0.7566, 0.6871, 0.6862],
            "ROC-AUC": [0.8337, 0.8339, 0.8262, 0.8247],
            "Recall (Severe)": [0.38, 0.79, 0.46, 0.46],
            "Precision (Severe)": [0.66, 0.44, 0.60, 0.59]
        }
        st.dataframe(pd.DataFrame(binary_data).style.highlight_max(subset=["Accuracy", "F1 (Severe)", "Balanced Acc", "ROC-AUC", "Recall (Severe)", "Precision (Severe)"], axis=0, color='#DAA520'))
        
        st.markdown("---")
        st.markdown("### 5.2 Winner: LightGBM + Class Weights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Recall (Severe)", "79%", "↑ from 38%")
        with col2:
            st.metric("⚖️ Balanced Accuracy", "75.7%", "Best")
        with col3:
            st.metric("📈 ROC-AUC", "83.4%", "Excellent")
        
        st.markdown("""
        **Why This Model Wins**:
        - ✅ **Dramatically improved recall**: 79% (vs 8% baseline)
        - ✅ **Maintains strong ROC-AUC**: 0.8339
        - ✅ **Avoids synthetic noise**: No SMOTE artifacts
        - ✅ **Simple & fast**: class_weight='balanced' parameter
        - ✅ **Deployment-ready**: Stable predictions, interpretable
        
        **Trade-off**: Higher false positives (errs on side of caution) - desirable for safety applications
        """)
        
        st.markdown("---")
        st.markdown("### 5.3 SHAP Explainability Analysis")
        
        st.markdown("""
        SHAP analysis on BorderlineSMOTE model revealed top predictive features:
        
        **Top 10 Features**:
        1. **age** - Older individuals → higher severity
        2. **lane_width** - Narrower lanes → more severe
        3. **collision_label** - Chain/frontal collisions → severe
        4. **surface_condition** - Wet/unknown → severe
        5. **manoeuvre_label** - Loss of control → severe
        6. **vehicle_group** - Bicycles/motorcycles → high risk
        7. **weather_group** - Adverse weather → severe
        8. **hour_group** - Night/early morning → worse outcomes
        9. **user_category** - Pedestrians → vulnerable
        10. **sex_label** - Gender patterns in severity
        
        **Feature Reduction Test**:
        - Trained model with top 40 SHAP features (from 85)
        - Result: Accuracy 78.6%, F1 0.41, Recall 35%
        - ❌ Not suitable for deployment (worse performance, misses severe cases)
        """)
        
        st.warning("⚠️ **Key Finding**: Reducing from 85 to 40 features decreased performance. All 85 features contribute to accuracy.")
    
    st.markdown("---")
    
    # Section 6: Final Comparison
    with st.expander("🏆 6. Final Model Comparison & Recommendation"):
        st.markdown("### Multiclass vs Binary")
        
        # Comparison table
        comparison_data = {
            "Metric": ["Accuracy", "Macro F1", "Balanced Accuracy", "Severe Recall", "ROC-AUC", "Deployment Ready"],
            "Multiclass (Borderline SMOTE)": ["64.3%", "50.1%", "49.5%", "22%", "N/A", "❌"],
            "Binary (Class Weights)": ["73.9%", "56.6%", "75.7%", "79%", "83.4%", "✅"]
        }
        st.table(pd.DataFrame(comparison_data))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🥇 Recommended: Binary Model
            
            **Best for Production**:
            - LightGBM with `class_weight='balanced'`
            - 85 features (all)
            - Trained on original (unbalanced) data
            
            **Use Cases**:
            - 🚑 Emergency triage
            - 📱 Real-time risk scoring
            - 🚨 Automated alerts
            - 🎯 Resource allocation
            
            **Strengths**:
            - High recall (catches 79% of severe cases)
            - Strong ROC-AUC (0.83)
            - Fast inference (<10ms)
            - Simple to maintain
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Alternative: Multiclass Model
            
            **Best for Analysis**:
            - LightGBM + Borderline SMOTE
            - 85 features
            - Predicts 4 severity levels
            
            **Use Cases**:
            - 📈 Policy research
            - 🧪 Injury pattern studies
            - 📝 Detailed reporting
            - 🎓 Academic analysis
            
            **Limitations**:
            - Lower recall for Killed class (22%)
            - More false negatives
            - Complex probability calibration
            """)
        
        st.success("""
        ### ✅ Final Recommendation
        
        **Deploy Binary Model** for real-time safety applications where catching severe cases is critical.
        
        **Use Multiclass Model** for retrospective analysis and detailed severity stratification.
        """)
    
    st.markdown("---")
    
    # Section 7: Key Takeaways
    st.subheader("💡 Key Takeaways")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Problem Framing Matters**
        - Binary classification significantly outperforms multiclass
        - Task simplification improved all metrics
        - Domain knowledge guides model selection
        """)
    
    with col2:
        st.markdown("""
        **⚖️ Handling Imbalance**
        - Class weights > SMOTE for binary
        - Borderline SMOTE > plain SMOTE for multiclass
        - Oversampling helps but adds complexity
        """)
    
    with col3:
        st.markdown("""
        **🚀 Deployment Strategy**
        - Simple models with good recall win
        - Interpretability enables stakeholder buy-in
        - Production readiness > perfect metrics
        """)
    
    st.markdown("---")
    
    # Section 8: Interactive Prediction Tool
    st.subheader("🔮 Try the Binary Prediction Model")
    
    # Try to load the model
    try:
        import joblib
        import pickle
        import numpy as np
        from pathlib import Path
        
        models_dir = Path("models")
        
        # Load model and preprocessors
        binary_model = joblib.load(models_dir / 'binary_lgbm_classweight.pkl')
        num_imputer = joblib.load(models_dir / 'num_imputer.pkl')
        cat_imputer = joblib.load(models_dir / 'cat_imputer.pkl')
        encoder = joblib.load(models_dir / 'onehot_encoder.pkl')
        scaler = joblib.load(models_dir / 'standard_scaler.pkl')
        
        with open(models_dir / 'feature_names_binary.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        with open(models_dir / 'column_info.pkl', 'rb') as f:
            column_info = pickle.load(f)
        
        st.success("✅ Model loaded successfully! Enter accident details below to predict severity.")
        
        with st.form("binary_prediction_form"):
            st.markdown("**Enter Accident Details:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=35, help="Age of person involved")
                vehicle_group = st.selectbox("Vehicle Type", 
                    ["Car", "Motorcycle", "Bicycle", "Heavy_vehicle", "Other"], key="vehicle")
                collision_label = st.selectbox("Collision Type",
                    ["Frontal_two_vehicles", "Side_collision", "Rear_end", "Chain_reaction", "No_collision"])
            
            with col2:
                lane_width = st.number_input("Lane Width (m)", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
                weather_group = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Fog"])
                surface_condition_label = st.selectbox("Surface Condition",
                    ["Normal", "Wet", "Snow_ice", "Other"])
            
            with col3:
                hour_group = st.selectbox("Time of Day",
                    ["Morning", "Afternoon", "Evening", "Night"])
                road_group = st.selectbox("Road Type", ["Highway", "Major_road", "Local_road"])
                is_urban = st.checkbox("Urban Area", value=True)
            
            st.markdown("**Safety Equipment:**")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                seatbelt_used = st.checkbox("Seatbelt Used", value=True)
            with col5:
                helmet_used = st.checkbox("Helmet Used", value=False)
            with col6:
                is_weekend = st.checkbox("Weekend", value=False)
            
            predict_btn = st.form_submit_button("🚨 Predict Severity", use_container_width=True)
            
            if predict_btn:
                try:
                    # Create input dataframe with all required features
                    input_data = {
                        'age': age,
                        'lane_width': lane_width,
                        'collision_label': collision_label,
                        'surface_condition_label': surface_condition_label,
                        'manoeuvre_label': 'Going_straight',
                        'sex_label': 'Male',
                        'user_category_label': 'Driver',
                        'seat_position_label': 'Front',
                        'journey_purpose_label': 'Commute',
                        'vehicle_group': vehicle_group,
                        'impact_group': 'Front',
                        'road_group': road_group,
                        'weather_group': weather_group,
                        'day_of_week': 'Saturday' if is_weekend else 'Monday',
                        'hour_group': hour_group,
                        'season': 2,
                        'age_group': 'Adult',
                        'is_weekend': is_weekend,
                        'is_holiday': False,
                        'seatbelt_used': seatbelt_used,
                        'helmet_used': helmet_used,
                        'any_protection_used': seatbelt_used or helmet_used,
                        'protection_effective': seatbelt_used or helmet_used,
                        'motorcycle_side_impact': vehicle_group == 'Motorcycle',
                        'is_night': hour_group == 'Night',
                        'is_urban': is_urban
                    }
                    
                    input_df = pd.DataFrame([input_data])
                    
                    # Preprocessing pipeline
                    input_df[column_info['numeric_cols']] = num_imputer.transform(input_df[column_info['numeric_cols']])
                    input_df[column_info['categorical_cols']] = cat_imputer.transform(input_df[column_info['categorical_cols']])
                    
                    X_encoded = encoder.transform(input_df[column_info['categorical_cols']])
                    X_scaled = scaler.transform(input_df[column_info['numeric_cols']])
                    
                    X_final = np.hstack([
                        X_scaled,
                        X_encoded,
                        input_df[column_info['binary_cols']].values
                    ])
                    
                    X_final_df = pd.DataFrame(X_final, columns=feature_names)
                    
                    # Predict
                    prob_severe = binary_model.predict_proba(X_final_df)[0, 1]
                    is_severe = prob_severe >= 0.5
                    
                    # Display results
                    st.markdown("---")
                    
                    if is_severe:
                        st.markdown("### 🚨 <span style='color: #e74c3c; font-weight: bold;'>HIGH RISK - SEVERE OUTCOME PREDICTED</span>", unsafe_allow_html=True)
                        st.error(f"**Probability of Severe Outcome: {prob_severe:.1%}**")
                        st.warning("⚠️ High likelihood of Hospitalization or Fatality")
                    else:
                        st.markdown("### ✅ <span style='color: #2ecc71; font-weight: bold;'>LOW RISK - NOT SEVERE</span>", unsafe_allow_html=True)
                        st.success(f"**Probability of Severe Outcome: {prob_severe:.1%}**")
                        st.info("Lower likelihood of serious injury")
                    
                    # Metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Severe Risk", f"{prob_severe:.1%}")
                    with col_m2:
                        st.metric("Not Severe", f"{1-prob_severe:.1%}")
                    with col_m3:
                        st.metric("Model Confidence", f"{max(prob_severe, 1-prob_severe):.1%}")
                    
                    # Risk factors
                    with st.expander("📊 Risk Factor Analysis"):
                        st.markdown("**High Risk Factors Present:**")
                        factors = []
                        if vehicle_group in ["Motorcycle", "Bicycle"]:
                            factors.append(f"- {vehicle_group}: Vulnerable road user (↑ injury risk)")
                        if not seatbelt_used and vehicle_group == "Car":
                            factors.append("- No seatbelt: Major injury risk multiplier")
                        if not helmet_used and vehicle_group == "Motorcycle":
                            factors.append("- No helmet: Extremely high fatality risk")
                        if hour_group == "Night":
                            factors.append("- Night time: Reduced visibility")
                        if collision_label in ["Frontal_two_vehicles", "Side_collision"]:
                            factors.append(f"- {collision_label}: High impact force")
                        if weather_group != "Clear":
                            factors.append(f"- {weather_group} weather: Reduced control")
                        
                        if factors:
                            for f in factors:
                                st.write(f)
                        else:
                            st.write("✅ No major risk factors identified")
                        
                        st.markdown("**Protective Factors:**")
                        protective = []
                        if is_urban:
                            protective.append("- Urban area: Lower speeds")
                        if seatbelt_used:
                            protective.append("- Seatbelt: ~50% reduction in severe injury")
                        if helmet_used and vehicle_group == "Motorcycle":
                            protective.append("- Helmet: ~40% reduction in head injury fatality")
                        if hour_group in ["Morning", "Afternoon"]:
                            protective.append("- Daylight hours: Better visibility")
                        
                        for p in protective:
                            st.write(p)
                        
                        st.caption("**Model**: LightGBM with class_weight='balanced' | **Accuracy**: 73.9% | **Recall (Severe)**: 79% | **ROC-AUC**: 83.4%")
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Please run the Bin_Modeling.ipynb notebook to export the model.")
        st.info("""
        **To enable predictions:**
        1. Open [Bin_Modeling.ipynb](Bin_Modeling.ipynb)
        2. Run all cells (especially the last export cell)
        3. Restart this Streamlit app
        """)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>        
        <p>Road Accidents in France - Data Science Project 2025</p>
        <p>Developed by Z. Malik and M. Peuyn</p>
    </div>
    """,
    unsafe_allow_html=True
)
