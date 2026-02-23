
from src.models.split_and_transform import split_and_transform, Columns
import src.models.test_models as tm

#### LOADING DATA FILE AND LIBRARIES
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    make_scorer
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import shap

import joblib
import pickle
from pathlib import Path

from sklearn.decomposition import PCA

class TestPlan:
    basic = False
    logistic = False
    lgbm = False



def get_master_df():
    # TODO: For experiments only. Use the real processed data.
    source_file = "tests/resources/reference_file_master_acc.csv"

    acc = pd.read_csv(
        source_file,
        encoding="latin-1",
        low_memory=False,
        index_col="accident_id"
    )
    return acc

def make_model(acc: pd.DataFrame):
    print("\n -- Splitting Data --")
    (
        X_train_final, X_test_final,
        y_train_enc, y_test_enc,
        label_encoder, encoder, num_imputer, cat_imputer,
        feature_names
    ) = split_and_transform(acc)

    if TestPlan.basic:
        print("\n -- Comparing basic models --")
        tm.compare_basic_models(X_train_final, y_train_enc)
    
    if TestPlan.logistic:
        print("\n -- Testing logistic regression --")
        tm.test_logistic_regression(X_train_final, y_train_enc)

    # Wrap X_train_lgbm into a DataFrame with names
    X_train_lgbm = pd.DataFrame(X_train_final, columns=feature_names)
    X_test_lgbm = pd.DataFrame(X_test_final, columns=feature_names)

    if TestPlan.lgbm:
        print("\n -- Testing LGBM --")
        tm.test_lgbm(X_train_lgbm, y_train_enc, X_test_lgbm, y_test_enc, feature_names)

    print("\n -- Testing smote LGBM --")
    (X_train_bs85, y_train_bs85, X_train_scaled) = tm.test_lgbm_smote(
        X_train_lgbm, y_train_enc, X_test_lgbm, feature_names)
    
    tm.train_final_model()

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

    # Scale original train and test sets
    scaler = StandardScaler()
    scaler.fit(X_train_final)
    X_test_scaled  = scaler.transform(X_test_final)

    # Fit PCA on original training set
    pca = PCA(n_components=2, random_state=42)

    pca.fit(X_train_scaled)
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

    explainer = shap.TreeExplainer(lgbm_final)
    shap_values = explainer.shap_values(X_test_lgbm)
    for c in range(4):
        print(f"\n=== SHAP Summary Plot for Class {c} ===")
        shap.summary_plot(shap_values[c], X_test_lgbm, show=False)

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

    ### Implement SHAP findings and retrain the model
    # Compute mean(|SHAP|) across classes (multiclass-safe)
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    # Selecting top 40 features
    top_40_idx = np.argsort(mean_shap)[-40:]
    top_40_features = [feature_names[i] for i in top_40_idx]

    print("Top 40 SHAP Features:")
    print(top_40_features)

    #ubset training and test sets
    X_test_top40  = X_test_lgbm[top_40_features]
    # Retrain LightGBM on reduced features and Evaluating results

    best_lgbm_reduced = LGBMClassifier(
        objective='multiclass',
        num_class=4,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        n_estimators=200,
        random_state=42
    )

    y_pred = best_lgbm_reduced.predict(X_test_top40)

    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_test_enc, y_pred)

    print("Reduced Model Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")


    # Create models directory
    Path("models").mkdir(exist_ok=True)

    # CRITICAL: Reconstruct feature_names with encoded categorical names
    # Get categorical feature names from the encoder (AFTER encoding)
    categorical_feature_names = encoder.get_feature_names_out(Columns.categorical)

    # Numeric + binary names
    numeric_feature_names = Columns.numeric
    binary_feature_names = Columns.binary

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
            'numeric_cols': Columns.numeric,
            'categorical_cols': Columns.categorical,
            'binary_cols': Columns.binary
        }, f)

    print("\n✅ Multiclass model and preprocessors exported successfully!")
    print("   Model: lgbm_bs85_final")
    print("   Location: models/")
    print(f"   Files created: {len(list(Path('models').glob('*.pkl')))} PKL files")

if __name__ == "__main__":
    acc = get_master_df()
    make_model(acc)

