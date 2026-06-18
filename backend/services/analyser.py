from pathlib import Path

def analyse_dataset(file_path: Path)-> Dict[str, Any]:
    """
    Returns a descriptive JSON report of the dataset
    """
    
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    row_count = max(len(df), 1)

    possible_targets = [col for col in df.columns if df[col].nunique() < 20]
    target_distributions = {}

    for col in possible_targets:
        target_distributions[col] = (
            df[col].value_counts(normalize=True).round(4).to_dict()
        )
    
    analysis = {
        "columns": list(df.columns),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / row_count * 100).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percentage": (df.duplicated().sum() / row_count * 100),
        "numerical_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "possible_target_columns": possible_targets,
        "target_distributions": target_distributions,
        "preview": df.head(5).to_dict(orient="records")
    }
    
    return analysis  


"""
extra stuff i can add
- correlations
- skewness
- outlier detection
- feature importance estimates
- target inference
"""