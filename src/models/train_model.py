
from src.models.split_and_transform import split_and_transform
from src.models.export import export
from imblearn.over_sampling import BorderlineSMOTE
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)
from pathlib import Path

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
    cr = classification_report(y_test, y_pred_bs85)
    print(cr)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_bs85)
    print(cm)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "classification_report": cr
    }

def apply_smote_transform(X_train, y_train, feature_names):
    # Borderline SMOTE Integration (bsmote85)
    print("  -- Nearest neighbors")
    nn_k = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    nn_m = NearestNeighbors(n_neighbors=3, n_jobs=-1)

    print("  -- Creating smote")
    counts = pd.Series(y_train).value_counts().sort_index()
    majority = counts.max()
    target = int(0.5 * majority)

    sampling_strategy = {
        cls: max(count, target)
        for cls, count in counts.items()
        if count < target
    }
    print("  -- Sampling strategy:", sampling_strategy)

    bsmote85 = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        kind="borderline-1",
        k_neighbors=nn_k,
        m_neighbors=nn_m
    )

    print("  -- Fit resample smote")
    X_train_bs85, y_train_bs85 = bsmote85.fit_resample(X_train, y_train)

    # Convert to DataFrame with feature names (required for LightGBM feature name matching)
    print("  -- Rebuild dataframe:")
    X_train_bs85 = pd.DataFrame(X_train_bs85, columns=feature_names)

    return (X_train_bs85, y_train_bs85)

def train_model_from_dataframe(accidents_df: pd.DataFrame, info, timestamp):
    print("\n -- Splitting Data --")
    (
        X_train_final, X_test_final,
        y_train_enc, y_test_enc,
        label_encoder, encoder, num_imputer, cat_imputer,
        feature_names
    ) = split_and_transform(accidents_df)

    # Wrap X_train_lgbm into a DataFrame with names
    X_train_lgbm = pd.DataFrame(X_train_final, columns=feature_names)
    X_test_lgbm = pd.DataFrame(X_test_final, columns=feature_names)

    print("\n -- Apply smote transform --")
    (X_train_bs85, y_train_bs85) = apply_smote_transform(X_train_lgbm, y_train_enc, feature_names)

    print("\n -- Train final model --")
    model = train_final_model(X_train_bs85, y_train_bs85)

    print("\n -- Evaluate final model --")
    metrics = evaluate_final_model(model, X_test_lgbm, y_test_enc)

    print("\n -- Export final model --")

    export(model, feature_names, label_encoder, encoder, cat_imputer, num_imputer, info, metrics, timestamp)

def train_model(nrows: int | None, info):
    import src.data.sql as sql

    folder = Path("data/processed")
    files = list(folder.glob("accidents_*.db"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly 1 file, found {len(files)}")
    file = files[0]
    timestamp = file.stem.split("_")[1]

    df = sql.read_accidents(file, nrows)
    train_model_from_dataframe(df, info, timestamp)

if __name__ == "__main__":
    params = {"nrows": 1_000_000}
    train_model(params["nrows"], params)