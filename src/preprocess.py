import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_give_me_some_credit(
    file_path: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
):
    """
    Preprocess Give Me Some Credit dataset.

    Target:
        SeriousDlqin2yrs
        1 = serious financial distress / high risk
        0 = low risk
    """
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be in (0, 1).")
    df = pd.read_csv(file_path)

    unnamed_cols = [col for col in df.columns if "Unnamed" in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    target_col = "SeriousDlqin2yrs"
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    feature_names = X.columns.tolist()

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_development,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    X_train = pd.DataFrame(X_train_imputed, columns=feature_names)
    X_val = pd.DataFrame(X_val_imputed, columns=feature_names)
    X_test = pd.DataFrame(X_test_imputed, columns=feature_names)

    clip_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]

    for col in clip_cols:
        if col in X_train.columns:
            lower = X_train[col].quantile(0.01)
            upper = X_train[col].quantile(0.99)
            X_train[col] = X_train[col].clip(lower, upper)
            X_val[col] = X_val[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.astype(np.float32)
    X_val_scaled = X_val_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)
    y_val = y_val.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        feature_names,
    )


def preprocess_german_credit_numeric(
    file_path: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
):
    """
    Preprocess German Credit numeric dataset.

    Original label:
        1 = good credit risk
        2 = bad credit risk

    Converted label:
        0 = good credit risk
        1 = bad credit risk
    """
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be in (0, 1).")
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.map({1: 0, 2: 1}).astype(int)

    feature_names = [f"x{i}" for i in range(X.shape[1])]

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_development,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.astype(np.float32)
    X_val_scaled = X_val_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)
    y_val = y_val.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        feature_names,
    )


def _read_default_credit_card_clients(file_path: str) -> pd.DataFrame:
    suffix = str(file_path).lower()
    if suffix.endswith((".xls", ".xlsx")):
        try:
            df = pd.read_excel(file_path, header=1)
        except ImportError as exc:
            raise ImportError(
                "Reading Default of Credit Card Clients from .xls requires "
                "`xlrd>=2.0.1`. Install project requirements first."
            ) from exc
    else:
        df = pd.read_csv(file_path)

    df = df.rename(columns=lambda col: str(col).strip())
    target_col = "default payment next month"
    if target_col not in df.columns and suffix.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path, header=0)
        df = df.rename(columns=lambda col: str(col).strip())

    if target_col not in df.columns:
        raise ValueError(
            "Default Credit target column not found. Expected "
            "'default payment next month'."
        )

    return df


def preprocess_default_credit_card_clients(
    file_path: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
):
    """
    Preprocess Default of Credit Card Clients dataset.

    Target:
        default payment next month
        1 = default / high risk
        0 = non-default / low risk
    """
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be in (0, 1).")

    df = _read_default_credit_card_clients(file_path)
    target_col = "default payment next month"

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    feature_names = X.columns.tolist()

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_development,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    X_train_scaled = X_train_scaled.astype(np.float32)
    X_val_scaled = X_val_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)
    y_val = y_val.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        feature_names,
    )


def print_dataset_summary(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    name="Dataset",
):
    print(f"===== {name} =====")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)
    print("Train positive ratio:", y_train.mean())
    print("Validation positive ratio:", y_val.mean())
    print("Test positive ratio:", y_test.mean())
    print("Any NaN in X_train:", np.isnan(X_train).any())
    print("Any NaN in X_val:", np.isnan(X_val).any())
    print("Any NaN in X_test:", np.isnan(X_test).any())
    print()
