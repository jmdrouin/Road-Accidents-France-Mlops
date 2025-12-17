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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", "839,985", help="Total number of accident records")
    with col2:
        st.metric("Features", "52", help="Number of features in dataset")
    with col3:
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
        st.markdown("### ⏰ Temporal Patterns")
        
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
            st.markdown("#### 🕐 Accidents by Time of Day")
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
            st.markdown("#### 🎉 Accident Likelihood: Holidays vs Regular Weekdays")
            
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
                    if len(accidents_by_hour) > 0:
                        peak_hour_val = int(accidents_by_hour.sort_values('count', ascending=False)['hour'].iloc[0])
                        st.metric("Peak Hour", f"{peak_hour_val:02d}:00")
                    else:
                        st.metric("Peak Hour", "N/A")
                with col3:
                    if len(accidents_by_month) > 0:
                        peak_month = accidents_by_month.sort_values('count', ascending=False)['month_name'].iloc[0]
                        st.metric("Peak Month", str(peak_month))
                    else:
                        st.metric("Peak Month", "N/A")
                with col4:
                    if len(accidents_by_weekday) > 0:
                        peak_weekday = accidents_by_weekday.sort_values('count', ascending=False)['day_of_week'].iloc[0]
                        st.metric("Peak Weekday", str(peak_weekday))
                    else:
                        st.metric("Peak Weekday", "N/A")
                    
        except Exception as e:
            st.error(f"Error loading temporal data: {str(e)}")
            st.info("Please ensure caracteristics.csv is in the correct location.")
    
    with tab2:
        st.markdown("### 🗺️ Geographical Distribution")
        
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
                    st.metric("Total Accidents", f"{len(df_all):,}")
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
        st.warning("🚧 **Coming Soon**: Distribution plots for key features.")
        st.write("Expected visualizations:")
        st.write("- Road type distribution")
        st.write("- Weather conditions")
        st.write("- Lighting conditions")
        st.write("- Vehicle types involved")
    
    with tab4:
        st.markdown("### ⚠️ Severity Analysis")
        st.warning("🚧 **Coming Soon**: Analysis of accident severity patterns.")
        st.write("Expected visualizations:")
        st.write("- Severity distribution")
        st.write("- Severity by time of day")
        st.write("- Severity by road type")
        st.write("- Severity correlation with weather")

# Page 2: Pre-processing & Feature Engineering
elif page == "Pre-processing & Feature engineering":
    st.markdown('<p class="section-header">🔧 Pre-processing & Feature Engineering</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Quality Section
    st.subheader("🧹 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🚧 **Work in Progress**: Missing values analysis")
        st.write("Will display:")
        st.write("- Missing value percentages by feature")
        st.write("- Visualization of missing data patterns")
        st.write("- Imputation strategies")
    
    with col2:
        st.info("🚧 **Work in Progress**: Outlier detection")
        st.write("Will display:")
        st.write("- Statistical outlier identification")
        st.write("- Box plots for numerical features")
        st.write("- Outlier treatment strategies")
    
    st.markdown("---")
    
    # Feature Engineering Section
    st.subheader("⚙️ Feature Engineering")
    
    tab1, tab2, tab3 = st.tabs(["New Features", "Encoding", "Scaling"])
    
    with tab1:
        st.markdown("### 🆕 Feature Creation")
        st.warning("🚧 **Coming Soon**: New engineered features")
        st.write("Planned features:")
        st.write("- **Temporal features**: Hour bins, Weekend flag, Rush hour flag")
        st.write("- **Interaction features**: Weather × Road condition, Time × Location")
        st.write("- **Aggregated features**: Historical accident counts by location")
        st.write("- **Geospatial features**: Distance to city center, Road density")
    
    with tab2:
        st.markdown("### 🔢 Categorical Encoding")
        st.warning("🚧 **Coming Soon**: Encoding strategies for categorical variables")
        st.write("Encoding methods:")
        st.write("- One-Hot Encoding for nominal features")
        st.write("- Label Encoding for ordinal features")
        st.write("- Target Encoding for high-cardinality features")
        st.write("- Frequency Encoding where appropriate")
    
    with tab3:
        st.markdown("### 📏 Feature Scaling")
        st.warning("🚧 **Coming Soon**: Feature normalization and standardization")
        st.write("Scaling techniques:")
        st.write("- StandardScaler for normally distributed features")
        st.write("- MinMaxScaler for bounded features")
        st.write("- RobustScaler for features with outliers")
    
    st.markdown("---")
    
    # Feature Selection Section
    st.subheader("🎯 Feature Selection")
    st.info("🚧 **Work in Progress**: Feature importance analysis and selection")
    st.write("Methods to be applied:")
    st.write("- Correlation analysis")
    st.write("- Feature importance from tree-based models")
    st.write("- Recursive Feature Elimination (RFE)")
    st.write("- SHAP values for feature contribution")

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
