from pathlib import Path

def preprocess_dataset(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocesses the dataset by handling missing values, encoding categorical columns,
    scaling numerical columns, and splitting the data into training and testing sets.

    Args:
        file_path (Path): The path to the dataset file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training and testing sets.
    """
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    df = df.drop_duplicates()
    
    df = df.fillna(df.mean(numeric_only=True))

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Split the data into training and testing sets
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test