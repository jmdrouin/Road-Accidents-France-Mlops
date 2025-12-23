#!/usr/bin/env python3
"""Replace old Modelling and Conclusion pages with new Modeling & Optimization page"""

# Read the original file
with open('streamlit-app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start of Page 3 (# Page 3: Modelling)
page3_start = content.find('# Page 3: Modelling')
if page3_start == -1:
    print("❌ Could not find '# Page 3: Modelling'")
    exit(1)

# Find the start of the footer (# Footer)
footer_start = content.find('# Footer', page3_start)
if footer_start == -1:
    print("❌ Could not find '# Footer' after Page 3")
    exit(1)

# Keep everything before Page 3
before_page3 = content[:page3_start]

# Keep the footer and everything after it
after_and_including_footer = content[footer_start:]

# Our new Modeling & Optimization page content
new_page_content = '''# Page 3: Modeling & Optimization
elif page == "Modeling & Optimization":
    st.markdown('<p class="section-header">🤖 Modeling & Optimization</p>', unsafe_allow_html=True)
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
        st.metric("Target Classes", "4")
    with col4:
        st.metric("Models Tested", "15+")
    
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
        st.dataframe(pd.DataFrame(bench_data).style.highlight_max(axis=0, color='lightgreen'))
        
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
            - Accuracy: 0.64
            - Macro F1: 0.50
            - Balanced Accuracy: 0.49
            
            **Killed class (2)**: F1 = 0.23 (↑ from 0.05)
            """)
        
        st.success("✅ **Best Multiclass Model**: Borderline SMOTE strikes best balance between fairness and precision")
        
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
            "Model": ["Baseline LGBM", "LGBM + SMOTE", "LGBM + Borderline SMOTE", "LGBM + Class Weights"],
            "Accuracy": [0.7849, 0.6816, 0.6933, 0.7394],
            "F1 (Severe)": [0.1394, 0.4742, 0.4969, 0.5657],
            "Balanced Acc": [0.5311, 0.6798, 0.6969, 0.7566],
            "ROC-AUC": [0.6821, 0.7510, 0.7652, 0.8339],
            "Recall (Severe)": [0.08, 0.51, 0.50, 0.79]
        }
        st.dataframe(pd.DataFrame(binary_data).style.highlight_max(axis=0, color='lightgreen'))
        
        st.markdown("---")
        st.markdown("### 5.2 Winner: LightGBM + Class Weights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Recall (Severe)", "79%", "↑ from 8%")
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
        - Trained model with top 40 SHAP features
        - Result: Accuracy ↑ (78%), Recall ↓ (28%)
        - ❌ Not suitable for deployment (misses severe cases)
        """)
        
        st.info("💡 **Insight**: SHAP excellent for interpretation, not for aggressive feature reduction")
    
    st.markdown("---")
    
    # Section 6: Final Comparison
    with st.expander("🏆 6. Final Model Comparison & Recommendation"):
        st.markdown("### Multiclass vs Binary")
        
        # Comparison table
        comparison_data = {
            "Metric": ["Accuracy", "Macro F1", "Balanced Accuracy", "Severe Recall", "Deployment Ready"],
            "Multiclass (Borderline SMOTE)": ["63.7%", "50.8%", "51.2%", "22%", "❌"],
            "Binary (Class Weights)": ["73.9%", "69.0%", "75.7%", "79%", "✅"]
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

'''

# Assemble the new file
new_content = before_page3 + new_page_content + "\n" + after_and_including_footer

# Write it back
with open('streamlit-app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Successfully replaced old pages with new Modeling & Optimization page")
print(f"   Original file: {len(content)} characters")
print(f"   New file: {len(new_content)} characters")
print(f"   Difference: {len(new_content) - len(content)} characters")
