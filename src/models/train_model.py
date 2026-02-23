
from src.models.split_and_transform import split_and_transform
from src.models.export import export
from imblearn.over_sampling import BorderlineSMOTE

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)
from lightgbm import LGBMClassifier

def get_master_df(source_file):
    acc = pd.read_csv(
        source_file,
        encoding="latin-1",
        low_memory=False,
        index_col="accident_id"
    )
    return acc

def train_final_model(X_train, y_train):
    ### TRAINING the final model (lgbm_bs85 on the full Borderline‑SMOTE dataset (X_train_bs85, y_train_bs85))

    # ------------------------------
    # Train final model on X_train_bs85, y_train_bs85
    # ------------------------------
    lgbm_bs85_final = LGBMClassifier(
        is_unbalance=True,
        random_state=42
    )
    lgbm_bs85_final.fit(X_train, y_train)
    return lgbm_bs85_final

def evaluate_final_model(model, X_test, y_test):
    # ------------------------------
    # Evaluate on test set
    # ------------------------------
    X_test_85 = X_test  

    y_pred_bs85 = model.predict(X_test_85)

    # Metrics
    acc = accuracy_score(y_test, y_pred_bs85)
    f1_macro = f1_score(y_test, y_pred_bs85, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_pred_bs85)

    print("\n=== Final Model: LightGBM (Controlled SMOTE, 85 features) ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"F1 Macro:          {f1_macro:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_bs85))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_bs85))

def apply_smote_transform(X_train, y_train, feature_names):
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

    return (X_train_bs85, y_train_bs85)

def train_model():
    # TODO: For experiments only. Use the real processed data.
    source_file = "tests/resources/reference_file_master_acc.csv"
    print("f\n -- Getting dataframe ({source_file}) --")
    acc = get_master_df(source_file)

    print("\n -- Splitting Data --")
    (
        X_train_final, X_test_final,
        y_train_enc, y_test_enc,
        label_encoder, encoder, num_imputer, cat_imputer,
        feature_names
    ) = split_and_transform(acc)

    # Wrap X_train_lgbm into a DataFrame with names
    X_train_lgbm = pd.DataFrame(X_train_final, columns=feature_names)
    X_test_lgbm = pd.DataFrame(X_test_final, columns=feature_names)

    print("\n -- Apply smote transform --")
    (X_train_bs85, y_train_bs85) = apply_smote_transform(X_train_lgbm, y_train_enc, feature_names)

    print("\n -- Train final model --")
    model = train_final_model(X_train_bs85, y_train_bs85)

    print("\n -- Evaluate final model --")
    evaluate_final_model(model, X_test_lgbm, y_test_enc)

    print("\n -- Export final model --")
    export(model, feature_names, label_encoder, encoder, cat_imputer, num_imputer)

if __name__ == "__main__":
    train_model()