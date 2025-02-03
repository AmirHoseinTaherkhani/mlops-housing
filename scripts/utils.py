import numpy as np
import pandas as pd

def preprocess_data(df):
    """Apply preprocessing pipeline to data (train/test)."""
    columns_to_drop1 = ['Id' ,'LotFrontage', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    columns_to_drop2 = ['GarageArea', 'TotRmsAbvGrd', '1stFlrSF']
    columns_to_drop3 = ['YearBuilt', 'YearRemodAdd', 'YrSold']
    columns_to_drop4 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']

    # Create Age Features BEFORE Dropping Year Columns
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # Combine all columns to drop
    cols_to_remove = columns_to_drop1 + columns_to_drop2 + columns_to_drop3 + columns_to_drop4

    # Drop columns
    for col in cols_to_remove:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True, errors='ignore')

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    new_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Return the modified DataFrame
    return new_df
