from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter

from sklearn.decomposition import PCA

def compare_basic_models(X_train, y_train):
    ### Performance Comparisons of top 5 Models with Benchmarking 
    # Running all five models separately in the order of quickest first → heaviest last 
    # by benchmarking all five models on a 200k sample

    #Create a 200k sample from your training set

    #Wrap your numpy array in a DataFrame
    X_df = pd.DataFrame(X_train)
    y_series = pd.Series(y_train)

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

def test_logistic_regression(X_train, y_train):
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

    out = cross_validate(logreg, X_train, y_train, cv=cv, scoring=scorers)

    print({
        "Model": "LogisticRegression",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })

def test_lgbm(X_train, y_train, X_test, y_test, feature_names):
    #### LGBM - X_train_final, y_train_final (_,85)

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
    lgbm.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorers = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy"
    }

    out = cross_validate(lgbm, X_train, y_train, cv=cv, scoring=scorers)

    print({
        "Model": "LightGBM",
        "Mean Accuracy": out["test_accuracy"].mean(),
        "Mean F1": out["test_f1_macro"].mean(),
        "Mean Balanced Acc": out["test_balanced_accuracy"].mean()
    })

    # Predict on test set
    y_pred = lgbm.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def test_lgbm_smote(X_train, y_train, X_test, feature_names):
    ### TRAINING LGBM (SMOTE & Borderline SMOTE) (85) - Comparison
    # SMOTE Integration
    # Apply SMOTE only on training set
    smote85 = SMOTE(random_state=42)

    X_train_s85, y_train_s85 = smote85.fit_resample(X_train, y_train)

    # Check new class distribution
    print("\nOriginal shape of X_train :", X_train.shape)
    print("\nSMOTE shape of X_train :", X_train_s85.shape)

    print("Original class distribution:", Counter(y_train))
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

    X_train_bs85, y_train_bs85 = bsmote85.fit_resample(X_train, y_train)

    # Convert to DataFrame with feature names (required for LightGBM feature name matching)
    X_train_bs85 = pd.DataFrame(X_train_bs85, columns=feature_names)

    print("\nOriginal shape of X_train :", X_train.shape)
    print("\nBorderline SMOTE shape of X_train :", X_train_bs85.shape)

    print("\nOriginal class distribution of y_train :", np.bincount(y_train))
    print("\nBorderline SMOTE class distribution of y_train :", np.bincount(y_train_bs85))
    # Training and Comparing LGBM on SMOTE and Borderline SMOTE for Original 85 Features

    ### Modeling Optimization with Feature Importance, PCA, SMOTE & Borderline SMOTE (60) 
    #### Extracting Feature Importance and Subsetting (85->60)
    #Training a quick LightGBM model and Extracting Feature Importance 

    lgb = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    lgb.fit(X_train, y_train)
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

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    X_train_sel = X_train_df[selected_feature_names]
    X_test_sel = X_test_df[selected_feature_names]

    print(X_train_sel.shape)
    print(X_test_sel.shape)

    #### Scaling and PCA Implementation (60->30)
    #Scaling the 60 selected features

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)

    # PCA IMPLEMENTATION

    # Fit PCA on training data only
    n_components = min(30, min(X_train_scaled.shape))
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # TODO: Can this go?
    # X_test_pca= pca.transform(X_test_scaled)

    pca.explained_variance_ratio_.sum()
    #### SMOTE and Borderline SMOTE on PCA (30)
    #SMOTE on PCA (30 Features)

    # SMOTE Integration

    # Apply SMOTE only on training set
    smote_pca = SMOTE(random_state=42)

    X_train_s30, y_train_s30 = smote_pca.fit_resample(X_train_pca, y_train)

    # Check new class distribution
    print("\n==================== SMOTE on PCA (30 Features) ====================")

    print("\nOriginal PCA shape of X_train :",X_train_pca.shape)
    print("\nPCA SMOTE shape of X_train :",X_train_s30.shape)

    print("\nOriginal class distribution:", Counter(y_train))
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


    X_train_bs30, y_train_bs30 = bsmote_pca.fit_resample(X_train_pca, y_train)

    print("\n==================== CONTROLLED-SMOTE on PCA (30 Features) ====================")

    print("\nOriginal PCA shape of X_train :",X_train_pca.shape)
    print("\nPCA CONTROLLED-SMOTE shape of X_train :",X_train_bs30.shape)

    print("\nOriginal class distribution of y_train :",np.bincount(y_train))
    print("\nPCA CONTROLLED-SMOTE class distribution of y_train :",np.bincount(y_train_bs30))


    #### Inverse transform on SMOTE and Borderline-SMOTE to original feature space (30->60)
    # Inverse transform SMOTE output back to original feature space
    X_train_s60 = pca.inverse_transform(X_train_s30)

    print(X_train_s60.shape)
    print(X_train_scaled.var(axis=0).mean())
    print(X_train_s60.var(axis=0).mean())

    # Build final DataFrames with feature names

    X_train_s60 = pd.DataFrame(X_train_s60, columns=selected_feature_names)

    # Inverse transform Borderline-SMOTE output back to original feature space
    X_train_bs60 = pca.inverse_transform(X_train_bs30)

    print(X_train_bs60.shape)
    print(X_train_scaled.var(axis=0).mean())
    print(X_train_bs60.var(axis=0).mean())

    # Build final DataFrames with feature names
    X_train_bs60 = pd.DataFrame(X_train_bs60, columns=selected_feature_names)
    # Inverse transform X_test to original feature space

    return (X_train_bs85, y_train_bs85, X_train_scaled)