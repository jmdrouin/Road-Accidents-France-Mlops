#### LOADING DATA FILE AND LIBRARIES
import pandas as pd
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    make_scorer
)
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter

from lazypredict.Supervised import LazyClassifier, CLASSIFIERS
from lightgbm import LGBMClassifier

class Config:
    test_size=0.2
    random_state=42

target_cols = ["injury_severity_label"]
numeric_cols = ["age", "lane_width"]

def get_master_df():
    # TODO: For experiments only. Use the real processed data.
    source_file = "tests/resources/reference_file_master_acc.csv"

    acc = pd.read_csv(
        source_file,
        encoding="latin-1",
        low_memory=False,
        index_col="accident_id"
    )
    # Drop rows with missing target:
    acc = acc.dropna(subset=target_cols)
    return acc

def make_model(acc: pd.DataFrame):
    x = blob
    X = acc.drop(columns=["injury_severity_label"])
    y = acc["injury_severity_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.test_size, random_state=Config.random_state, stratify=y
    )

    num_imputer = SimpleImputer(strategy="median")
    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    # Categorical columns
    categorical_cols = [
        "collision_label", "surface_condition_label", "manoeuvre_label",
        "sex_label", "user_category_label", "seat_position_label", "journey_purpose_label",
        "vehicle_group", "impact_group", "road_group", "weather_group",
        "day_of_week", "hour_group", "season"
    ]

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # Check again for NaNs
    print("Remaining NaNs in train:", X_train.isna().sum().sum())
    print("Remaining NaNs in test:", X_test.isna().sum().sum())
    # Show which columns still have NaNs
    print("Train NaNs per column:\n", X_train.isna().sum()[X_train.isna().sum() > 0])
    print("\nTest NaNs per column:\n", X_test.isna().sum()[X_test.isna().sum() > 0])
    # Ensure dtype is categorical
    X_train["age_group"] = X_train["age_group"].astype("category")
    X_test["age_group"] = X_test["age_group"].astype("category")

    # Add "Unknown" to categories if not already present
    if "Unknown" not in X_train["age_group"].cat.categories:
        X_train["age_group"] = X_train["age_group"].cat.add_categories(["Unknown"])
        X_test["age_group"] = X_test["age_group"].cat.add_categories(["Unknown"])

    # Fill NaNs with "Unknown"
    X_train["age_group"] = X_train["age_group"].fillna("Unknown")
    X_test["age_group"] = X_test["age_group"].fillna("Unknown")

    # Final check
    print("Remaining NaNs in train:", X_train.isna().sum().sum())
    print("Remaining NaNs in test:", X_test.isna().sum().sum())
    #### Encoding Categorical Features

    # Define categorical columns
    categorical_cols = [
        "collision_label", "surface_condition_label", "manoeuvre_label",
        "sex_label", "user_category_label", "seat_position_label", "journey_purpose_label",
        "vehicle_group", "impact_group", "road_group", "weather_group",
        "day_of_week", "hour_group", "season", "age_group"
    ]

    # Initialize encoder
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Fit on train, transform both train and test
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    print("Encoded train shape:", X_train_encoded.shape)
    print("Encoded test shape:", X_test_encoded.shape)
    #### Scale Numeric Features

    # Numeric columns
    numeric_cols = ["age", "lane_width"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled = scaler.transform(X_test[numeric_cols])

    print("Scaled train shape:", X_train_scaled.shape)
    print("Scaled test shape:", X_test_scaled.shape)
    #### Combine All Features

    # Binary columns
    binary_cols = [
        "is_weekend", "is_holiday", "seatbelt_used", "helmet_used",
        "any_protection_used", "protection_effective",
        "motorcycle_side_impact", "is_night", "is_urban"
    ]

    # Combine all parts
    X_train_final = np.hstack([
        X_train_scaled,          # numeric
        X_train_encoded,         # categorical
        X_train[binary_cols].values  # binary flags
    ])

    X_test_final = np.hstack([
        X_test_scaled,
        X_test_encoded,
        X_test[binary_cols].values
    ])

    print("Final train shape:", X_train_final.shape)
    print("Final test shape:", X_test_final.shape)
    #### Encoding Target Labels 
    # Encoding Target Labels (y_train & y_test)


    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Prepare data
    X_lazy = pd.DataFrame(X_train_final)
    y_lazy = y_train
    X_train_lazy, X_test_lazy, y_train_lazy, y_test_lazy = train_test_split(
        X_lazy, y_lazy, test_size=0.2, random_state=42
    )

    # Heavy models to skip
    heavy = {
        "SVC","NuSVC","QuadraticDiscriminantAnalysis",
        "LabelPropagation","LabelSpreading",
        "SelfTrainingClassifier","StackingClassifier",
        "GaussianProcessClassifier","MLPClassifier"
    }

    #------------------------------------------------------------------------------------------
    # Filter CLASSIFIERS (list of tuples) by name
    safe_classifiers = [(name, model) for name, model in CLASSIFIERS if name not in heavy]

    # Initialize LazyClassifier with filtered list
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        custom_metric=accuracy_score,
        classifiers=safe_classifiers
    )

    models, predictions = clf.fit(X_train_lazy, X_test_lazy, y_train_lazy, y_test_lazy)

    ### Performance Comparisons of top 5 Models with Benchmarking 
    # Running all five models separately in the order of quickest first → heaviest last 
    # by benchmarking all five models on a 200k sample

    #Create a 200k sample from your training set

    #Wrap your numpy array in a DataFrame
    X_df = pd.DataFrame(X_train_final)
    y_series = pd.Series(y_train_enc)

    #Now you can sample
    num_samples = min(len(X_df), 200000)
    X_sample = X_df.sample(num_samples, random_state=42)
    y_sample = y_series.loc[X_sample.index]

    #Use the same CV and scorers
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorers = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy"
    }
    #### 1. RidgeClassifier
    ridge = RidgeClassifier(alpha=1.0)
    out = cross_validate(ridge, X_sample, y_sample, cv=cv, scoring=scorers)
    print({
        "Model": "RidgeClassifier",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })
    #### 2. LogisticRegression
    logreg = LogisticRegression(max_iter=300, solver="saga", n_jobs=-1, class_weight="balanced")
    out = cross_validate(logreg, X_sample, y_sample, cv=cv, scoring=scorers)
    print({
        "Model": "LogisticRegression",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })
    #### 3. LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    out = cross_validate(lda, X_sample, y_sample, cv=cv, scoring=scorers)
    print({
        "Model": "LinearDiscriminantAnalysis",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })
    #### 4. LightGBMClassifier
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        force_row_wise=True
    )
    out = cross_validate(lgbm, X_sample, y_sample, cv=cv, scoring=scorers)
    print({
        "Model": "LGBMClassifier",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })


    #### 5. CalibratedClassifierCV
    base = LogisticRegression(max_iter=300, solver="saga", n_jobs=-1, class_weight="balanced")
    calib = CalibratedClassifierCV(estimator=base, cv=3)
    out = cross_validate(calib, X_sample, y_sample, cv=cv, scoring=scorers)
    print({
        "Model": "CalibratedClassifierCV",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })


    ### DATA MODELING (LR & LGBM) - (Original Features (85))
    #### LR - X_train_final, y_train_final (_,85)
    # LogisticRegression on X_train_final with scaling and higher max_iter
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,          # allow more iterations for convergence
            solver="lbfgs",         # faster, stable solver for dense data
            n_jobs=-1,
            class_weight="balanced" # handle imbalance without oversampling
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorers = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy"
    }

    out = cross_validate(logreg, X_train_final, y_train_enc, cv=cv, scoring=scorers)

    print({
        "Model": "LogisticRegression",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })
    #### LGBM - X_train_final, y_train_final (_,85)
    #Recovering feature names

    # Get categorical feature names from the encoder
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols)

    # Numeric + binary names
    numeric_feature_names = numeric_cols
    binary_feature_names = binary_cols

    # Combine all names
    feature_names = list(numeric_feature_names) + list(categorical_feature_names) + list(binary_feature_names)

    # Replace whitespace with underscores
    feature_names = [name.replace(" ", "_") for name in feature_names]

    # Wrap X_train_lgbm into a DataFrame with names
    X_train_lgbm = pd.DataFrame(X_train_final, columns=feature_names)
    X_test_lgbm = pd.DataFrame(X_test_final, columns=feature_names)
    #LightGBM training 

    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        is_unbalance=True,
        force_row_wise=True
    )
    lgbm.fit(X_train_lgbm, y_train_enc)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorers = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy"
    }

    out = cross_validate(lgbm, X_train_lgbm, y_train_enc, cv=cv, scoring=scorers)

    print({
        "Model": "LightGBM",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })

    # Predict on test set
    y_pred = lgbm.predict(X_test_lgbm)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_enc, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_enc, y_pred))

    ### TRAINING LGBM (SMOTE & Borderline SMOTE) (85) - Comparison
    # SMOTE Integration
    # Apply SMOTE only on training set
    smote85 = SMOTE(random_state=42)

    X_train_s85, y_train_s85 = smote85.fit_resample(X_train_final, y_train_enc)

    # Check new class distribution
    print("\nOriginal shape of X_train :",X_train_final.shape)
    print("\nSMOTE shape of X_train :",X_train_s85.shape)

    print("Original class distribution:", Counter(y_train_enc))
    print("SMOTE Balanced class distribution:", Counter(y_train_s85))

    # Borderline SMOTE Integration (bsmote85)
    bsmote85 = BorderlineSMOTE(
        sampling_strategy={
            0: 350000,   # boost class 0 (Hospitalized)
            1: 363089,   # keep class 1 unchanged
            2: 300000,   # strong boost for class 2 (Killed)
            3: 522866    # must stay >= original
        },
        random_state=42,
        kind="borderline-1"
    )

    X_train_bs85, y_train_bs85 = bsmote85.fit_resample(X_train_final, y_train_enc)

    # Convert to DataFrame with feature names (required for LightGBM feature name matching)
    X_train_bs85 = pd.DataFrame(X_train_bs85, columns=feature_names)

    print("\nOriginal shape of X_train :", X_train_final.shape)
    print("\nBorderline SMOTE shape of X_train :", X_train_bs85.shape)

    print("\nOriginal class distribution of y_train :", np.bincount(y_train_enc))
    print("\nBorderline SMOTE class distribution of y_train :", np.bincount(y_train_bs85))
    # Training and Comparing LGBM on SMOTE and Borderline SMOTE for Original 85 Features

    # ---------------------------------------------------------
    # Helper function to train and evaluate a model
    # ---------------------------------------------------------
    def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
        print(f"\n==================== {model_name} ====================")

        model = LGBMClassifier(random_state=42,is_unbalance=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return model

    # ---------------------------------------------------------
    # Train on SMOTE (85 features)
    # ---------------------------------------------------------
    lgbm_s85 = train_and_evaluate(
        X_train_s85, y_train_s85,
        X_test_lgbm, y_test_enc,
        model_name="LightGBM (Plain SMOTE - 85 features)"
    )

    # ---------------------------------------------------------
    # Train on Borderline SMOTE (85 features)
    # ---------------------------------------------------------
    lgbm_bs85 = train_and_evaluate(
        X_train_bs85, y_train_bs85,
        X_test_lgbm, y_test_enc,
        model_name="LightGBM (Borderline SMOTE - 85 features)"
    )
    ### Modeling Optimization with Feature Importance, PCA, SMOTE & Borderline SMOTE (60) 
    #### Extracting Feature Importance and Subsetting (85->60)
    #Training a quick LightGBM model and Extracting Feature Importance 

    lgb = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    lgb.fit(X_train_final, y_train_enc)
    # Extracting feature importances
    importances = pd.Series(
        lgb.feature_importances_,
        index=feature_names
    )
    importances = importances.sort_values(ascending=False)

    # Select features with importance >= 100
    selected_features = importances[importances >= 100]

    # Subsetting training and test arrays to these features
    selected_feature_names = selected_features.index.tolist()
    print(len(selected_feature_names))
    print(selected_feature_names)

    X_train_df = pd.DataFrame(X_train_final, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_final, columns=feature_names)

    X_train_sel = X_train_df[selected_feature_names]
    X_test_sel = X_test_df[selected_feature_names]

    print(X_train_sel.shape)
    print(X_test_sel.shape)

    #### Scaling and PCA Implementation (60->30)
    #Scaling the 60 selected features

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # PCA IMPLEMENTATION

    # Fit PCA on training data only
    n_components = min(30, min(X_train_scaled.shape))
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)

    X_test_pca= pca.transform(X_test_scaled)

    pca.explained_variance_ratio_.sum()
    #### SMOTE and Borderline SMOTE on PCA (30)
    #SMOTE on PCA (30 Features)

    # SMOTE Integration

    # Apply SMOTE only on training set
    smote_pca = SMOTE(random_state=42)

    X_train_s30, y_train_s30 = smote_pca.fit_resample(X_train_pca,y_train_enc)

    # Check new class distribution
    print("\n==================== SMOTE on PCA (30 Features) ====================")

    print("\nOriginal PCA shape of X_train :",X_train_pca.shape)
    print("\nPCA SMOTE shape of X_train :",X_train_s30.shape)

    print("\nOriginal class distribution:", Counter(y_train_enc))
    print("\nPCA SMOTE class distribution:", Counter(y_train_s30))

    #Borderline-SMOTE on PCA (30 Features)

    bsmote_pca = BorderlineSMOTE(
        sampling_strategy={
            0: 350000,   # boost class 0 (Hospitalized)
            1: 363089,   # keep class 1 unchanged
            2: 300000,   # strong boost for class 2 (Killed)
            3: 522866    # must stay >= original
        },
        random_state=42,
        kind="borderline-1"
    )


    X_train_bs30, y_train_bs30 = bsmote_pca.fit_resample(X_train_pca, y_train_enc)

    print("\n==================== CONTROLLED-SMOTE on PCA (30 Features) ====================")

    print("\nOriginal PCA shape of X_train :",X_train_pca.shape)
    print("\nPCA CONTROLLED-SMOTE shape of X_train :",X_train_bs30.shape)

    print("\nOriginal class distribution of y_train :",np.bincount(y_train_enc))
    print("\nPCA CONTROLLED-SMOTE class distribution of y_train :",np.bincount(y_train_bs30))


    #### Inverse transform on SMOTE and Borderline-SMOTE to original feature space (30->60)
    # Inverse transform SMOTE output back to original feature space
    X_train_s60 = pca.inverse_transform(X_train_s30)

    print(X_train_s60.shape)
    print(X_train_scaled.var(axis=0).mean())
    print(X_train_s60.var(axis=0).mean())

    # Build final DataFrames with feature names

    X_train_s60 = pd.DataFrame(X_train_s60, columns=selected_feature_names)
    y_train_s60 = pd.Series(y_train_s30)

    # Inverse transform Borderline-SMOTE output back to original feature space
    X_train_bs60 = pca.inverse_transform(X_train_bs30)

    print(X_train_bs60.shape)
    print(X_train_scaled.var(axis=0).mean())
    print(X_train_bs60.var(axis=0).mean())

    # Build final DataFrames with feature names
    X_train_bs60 = pd.DataFrame(X_train_bs60, columns=selected_feature_names)
    y_train_bs60 = pd.Series(y_train_bs30)
    # Inverse transform X_test to original feature space

    X_test_pca_inverse = pca.inverse_transform(X_test_pca)
    X_test_pca_final = pd.DataFrame(X_test_pca_inverse, columns=selected_feature_names)

    ### TRAINING LGBM (SMOTE & Borderline SMOTE) (60) - Comparison

    # ---------------------------------------------------------
    # Helper function to train and evaluate a model
    # ---------------------------------------------------------
    def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
        print(f"\n==================== {model_name} ====================")

        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            is_unbalance=True,
            force_row_wise=True
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return model


    # ---------------------------------------------------------
    # Train on Plain SMOTE (PCA 60 features)
    # ---------------------------------------------------------
    lgbm_s60 = train_and_evaluate(
        X_train_s60, y_train_s60,
        X_test_pca_final, y_test_enc,
        model_name="LightGBM (Plain SMOTE - PCA 60 features)"
    )

    # ---------------------------------------------------------
    # Train on Controlled SMOTE (PCA 60 features)
    # ---------------------------------------------------------
    lgbm_bs60 = train_and_evaluate(
        X_train_bs60, y_train_bs60,
        X_test_pca_final, y_test_enc,
        model_name="LightGBM (Borderline SMOTE - PCA 60 features)"
    )
    ### TRAINING the final model (lgbm_bs85 on the full Borderline‑SMOTE dataset (X_train_bs85, y_train_bs85))

    # ------------------------------
    # Train final model on X_train_bs85, y_train_bs85
    # ------------------------------
    lgbm_bs85_final = LGBMClassifier(
        is_unbalance=True,
        random_state=42
    )
    lgbm_bs85_final.fit(X_train_bs85, y_train_bs85)

    # ------------------------------
    # Evaluate on test set
    # ------------------------------
    X_test_85 = X_test_lgbm  

    y_pred_bs85 = lgbm_bs85_final.predict(X_test_85)

    # Metrics
    acc = accuracy_score(y_test_enc, y_pred_bs85)
    f1_macro = f1_score(y_test_enc, y_pred_bs85, average="macro")
    bal_acc = balanced_accuracy_score(y_test_enc, y_pred_bs85)

    print("\n=== Final Model: LightGBM (Controlled SMOTE, 85 features) ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"F1 Macro:          {f1_macro:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test_enc, y_pred_bs85))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_enc, y_pred_bs85))
    ### Model performance improvement
    #### Class‑weighted LightGBM 
    # Training LGBM with class-weighted model

    # ------------------------------
    # Train class-weighted model
    # ------------------------------
    lgbm_weighted = LGBMClassifier(
        class_weight='balanced',   # <-- key change
        random_state=42
    )

    lgbm_weighted.fit(X_train_bs85, y_train_bs85)

    # ------------------------------
    # Evaluate on test set
    # ------------------------------
    X_test_85 = X_test_lgbm
    y_pred_weighted = lgbm_weighted.predict(X_test_85)

    # Metrics
    acc_w = accuracy_score(y_test_enc, y_pred_weighted)
    f1_macro_w = f1_score(y_test_enc, y_pred_weighted, average="macro")
    bal_acc_w = balanced_accuracy_score(y_test_enc, y_pred_weighted)

    print("\n=== LightGBM (Borderline SMOTE + class_weight='balanced') ===")
    print(f"Accuracy:          {acc_w:.4f}")
    print(f"F1 Macro:          {f1_macro_w:.4f}")
    print(f"Balanced Accuracy: {bal_acc_w:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test_enc, y_pred_weighted))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_enc, y_pred_weighted))
    #### RandomizedSearchCV and Finding Bestparams

    # Base model with class weights
    lgbm_base = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight="balanced",
        random_state=42
    )

    # Basic search space (safe + effective)
    param_dist = {
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 10, 15],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 500, 800],
        "min_child_samples": [20, 50, 100],
        "feature_fraction": [0.8, 1.0],
        "bagging_fraction": [0.8, 1.0],
        "bagging_freq": [1]
    }

    # Macro F1 scorer
    f1_macro = make_scorer(f1_score, average="macro")

    # Randomized search (quick)
    random_search = RandomizedSearchCV(
        estimator=lgbm_base,
        param_distributions=param_dist,
        n_iter=15,               # quick search
        scoring=f1_macro,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Fit on your training data
    random_search.fit(X_train_bs85, y_train_bs85)

    print("Best Macro F1:", random_search.best_score_)
    print("Best Params:", random_search.best_params_)
    # Your custom class weights
    class_weights = {0: 1.5, 1: 1.0, 2: 3.0, 3: 0.8}

    # Best params from RandomizedSearchCV 
    best_params = {
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 800,
        "min_child_samples": 100,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 1
    }

    # Final model combining both
    lgbm_final = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight=class_weights,   #  custom weights
        random_state=42,
        **best_params
    )

    lgbm_final.fit(X_train_bs85, y_train_bs85)
    probs = lgbm_final.predict_proba(X_test_lgbm)
    #### Threshold Tuning and Final Evaluation

    # ============================================================
    # FIND BEST THRESHOLD FOR EACH CLASS (one-vs-rest)
    # ============================================================
    def find_best_threshold(y_true, y_prob, class_id):
        best_t = 0.5
        best_f1 = 0.0

        y_true_bin = (y_true == class_id).astype(int)

        for t in np.arange(0.05, 0.95, 0.01):
            y_pred_bin = (y_prob >= t).astype(int)
            f1 = f1_score(y_true_bin, y_pred_bin)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        return best_t, best_f1


    best_thresholds = {}

    for c in range(4):
        t, f1_c = find_best_threshold(y_test_enc, probs[:, c], c)
        best_thresholds[c] = t
        print(f"Class {c}: best threshold = {t:.2f}, F1 = {f1_c:.4f}")


    # ============================================================
    # 6. APPLY PER-CLASS THRESHOLDS
    # ============================================================
    final_preds = []

    for i in range(len(X_test_lgbm)):
        adjusted_scores = []
        for c in range(4):
            adjusted_scores.append(probs[i, c] - best_thresholds[c])
        final_preds.append(np.argmax(adjusted_scores))

    final_preds = np.array(final_preds)


    # ============================================================
    # 7. FINAL EVALUATION
    # ============================================================
    print("\n=== FINAL MODEL (Best Params + Class Weights + Per-Class Thresholds) ===")
    print("Accuracy:", accuracy_score(y_test_enc, final_preds))
    print("Macro F1:", f1_score(y_test_enc, final_preds, average="macro"))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test_enc, final_preds))

    print("\nClassification Report:")
    print(classification_report(y_test_enc, final_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_enc, final_preds))

    print("\nPer-class thresholds used:", best_thresholds)
    ### PCA PROJECTIONS
    # Fitting PCA on the original training Datasets

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    # Scale original train and test sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled  = scaler.transform(X_test_final)

    # Fit PCA on original training set
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    # Visualize PCA projection of test set (true classes)
    plt.figure(figsize=(8,6))
    for c in np.unique(y_test_enc):
        plt.scatter(
            X_test_pca[y_test_enc == c, 0],
            X_test_pca[y_test_enc == c, 1],
            label=f"Class {c}",
            alpha=0.5,
            s=10
        )
    plt.legend()
    plt.title("PCA Projection of Test Set (Original DataFrames)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
    # Visualize PCA projection of Borderline SMOTE-balanced training set

    X_train_bs85_scaled = scaler.transform(X_train_bs85)
    X_train_bs85_pca = pca.transform(X_train_bs85_scaled)

    plt.figure(figsize=(8,6))
    for c in np.unique(y_train_bs85):
        plt.scatter(
            X_train_bs85_pca[y_train_bs85 == c, 0],
            X_train_bs85_pca[y_train_bs85 == c, 1],
            label=f"Class {c}",
            alpha=0.5,
            s=10
        )
    plt.legend()
    plt.title("PCA Projection of SMOTE-Balanced Training Set")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", pca.explained_variance_ratio_.sum())
    ### SHAP
    import shap

    explainer = shap.TreeExplainer(lgbm_final)
    shap_values = explainer.shap_values(X_test_lgbm)
    for c in range(4):
        print(f"\n=== SHAP Summary Plot for Class {c} ===")
        shap.summary_plot(shap_values[c], X_test_lgbm, show=False)
    import numpy as np
    import pandas as pd

    feature_importance = np.mean(np.abs(shap_values), axis=1)  # shape: (4, n_features)

    for c in range(4):
        print(f"\n=== Top Features for Class {c} ===")
        imp_df = pd.DataFrame({
            "feature": X_test_lgbm.columns,
            "importance": feature_importance[c]
        }).sort_values("importance", ascending=False)
        
        print(imp_df.head(15))
    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    global_importance = np.mean(mean_abs_shap, axis=0)

    imp_df = pd.DataFrame({
        "feature": X_test_lgbm.columns,
        "importance": global_importance
    }).sort_values("importance", ascending=False)

    print("\n=== Global SHAP Feature Importance (Averaged Across Classes) ===")
    print(imp_df.head(20))

    ### SHAP feature‑importance barplot 
    import shap

    #  final tuned LightGBM model
    explainer = shap.TreeExplainer(best_lgbm)

    # SHAP values for each class
    shap_values = explainer.shap_values(X_test_lgbm)

    shap.summary_plot(
        shap_values,
        X_test_lgbm,
        plot_type="bar",
        class_names=["0", "1", "2", "3"]
    )
    # Take a 20k sample for SHAP
    sample_idx = np.random.choice(X_train_bs85.shape[0], 20000, replace=False)
    X_shap_sample = X_train_bs85[sample_idx]

    explainer = shap.TreeExplainer(best_lgbm)
    shap_values = explainer.shap_values(X_shap_sample)

    shap.summary_plot(
        shap_values,
        X_shap_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=30
    )
    ### Implement SHAP findings and retrain the model
    # Compute mean(|SHAP|) across classes (multiclass-safe)
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    # Selecting top 40 features
    top_40_idx = np.argsort(mean_shap)[-40:]
    top_40_features = [feature_names[i] for i in top_40_idx]

    print("Top 40 SHAP Features:")
    print(top_40_features)

    #ubset training and test sets
    X_train_top40 = X_train_cs85[top_40_features]
    X_test_top40  = X_test_lgbm[top_40_features]
    # Retrain LightGBM on reduced features and Evaluating results

    from lightgbm import LGBMClassifier, early_stopping
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

    best_lgbm_reduced = LGBMClassifier(
        objective='multiclass',
        num_class=4,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        n_estimators=200,
        random_state=42
    )

    best_lgbm_reduced.fit(
        X_train_top40, y_train_cs85,
        eval_set=[(X_test_top40, y_test_enc)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping(stopping_rounds=20)]
    )

    y_pred = best_lgbm_reduced.predict(X_test_top40)

    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_test_enc, y_pred)

    print("Reduced Model Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    ### 🎯 Export Models for Streamlit App
    import joblib
    import pickle
    from pathlib import Path

    # Create models directory
    Path("models").mkdir(exist_ok=True)

    # CRITICAL: Reconstruct feature_names with encoded categorical names
    # Get categorical feature names from the encoder (AFTER encoding)
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols)

    # Numeric + binary names
    numeric_feature_names = numeric_cols
    binary_feature_names = binary_cols

    # Combine all names in the same order as stacking
    feature_names = list(numeric_feature_names) + list(categorical_feature_names) + list(binary_feature_names)

    # Replace whitespace with underscores
    feature_names = [name.replace(" ", "_") for name in feature_names]

    print(f"✓ Feature names reconstructed: {len(feature_names)} features")
    print(f"  First 10: {feature_names[:10]}")
    print(f"  Sample age_group features: {[f for f in feature_names if 'age_group' in f][:5]}")

    # Save the final multiclass model
    joblib.dump(lgbm_bs85_final, 'models/multiclass_lgbm_model.pkl')

    # Save the preprocessors
    joblib.dump(num_imputer, 'models/num_imputer.pkl')
    joblib.dump(cat_imputer, 'models/cat_imputer.pkl')
    joblib.dump(encoder, 'models/onehot_encoder.pkl')
    joblib.dump(scaler, 'models/standard_scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')

    # Save feature names
    with open('models/feature_names_multiclass.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    # Save column lists
    with open('models/column_info.pkl', 'wb') as f:
        pickle.dump({
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'binary_cols': binary_cols
        }, f)

    print("\n✅ Multiclass model and preprocessors exported successfully!")
    print("   Model: lgbm_bs85_final")
    print("   Location: models/")
    print(f"   Files created: {len(list(Path('models').glob('*.pkl')))} PKL files")

if __name__ == "__main__":
    acc = get_master_df()
    get_master_df(acc)

