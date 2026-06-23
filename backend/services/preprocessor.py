import os
import re
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Tuple, Any
import pandas as pd
import numpy as np

def preprocess_dataset(file_path: Path, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    Preprocesses the dataset by handling missing values, encoding categorical columns,
    scaling numerical columns, and splitting the data into training and testing sets.
    Uses ColumnTransformer and Pipeline to ensure zero column mismatch and handles unseen categories.

    Args:
        file_path (Path): The path to the dataset file.
        target_column (str): The target column to be predicted

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]: Training features, test features, train targets, test targets, and the fitted preprocessor.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Auto-detect and exclude high-cardinality identifiers/indices from training features
    total_rows = max(len(X), 1)
    ID_KEYWORDS = {
        "id",
        "uuid",
        "guid",
        "email",
        "phone",
        "mobile",
        "account",
        "customer",
        "member",
        "transaction",
        "user"
    }
    
    ignore_cols = []
    for col in X.columns:
        unique_count = X[col].nunique(dropna=True)
        ratio = unique_count / total_rows
        
        col_lower = col.lower()
        has_id_keyword = any(
            keyword in col_lower
            for keyword in ID_KEYWORDS
        )
        
        is_object = pd.api.types.is_object_dtype(X[col])
        
        if (
            (has_id_keyword and ratio > 0.50)
            or
            (is_object and ratio > 0.95)
        ):
            ignore_cols.append(col)
            
    X_clean = X.drop(columns=ignore_cols)
    
    numeric_cols = X_clean.select_dtypes(include=["number"]).columns
    categorical_cols = X_clean.select_dtypes(include=["object", "category"]).columns

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

    # Build production-grade pipeline
    transformers = []
    if len(numeric_cols) > 0:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, list(numeric_cols)))

    if len(categorical_cols) > 0:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, list(categorical_cols)))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    preprocessor.set_output(transform="pandas")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def save_preprocessor(preprocessor: Any, path: str):
    """
    Saves the fitted preprocessor pipeline to disk using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)

def load_preprocessor(path: str) -> Any:
    """
    Loads a fitted preprocessor pipeline from disk using joblib.
    """
    return joblib.load(path)