import numpy as np
import pandas as pd

def preprocess_data(df, train_features=None):
    """Apply preprocessing pipeline to data (train/test) and ensure consistency in test data."""
    
    columns_to_drop1 = ['Id', 'LotFrontage', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    columns_to_drop2 = ['GarageArea', 'TotRmsAbvGrd', '1stFlrSF']
    columns_to_drop3 = ['YearBuilt', 'YearRemodAdd', 'YrSold']
    columns_to_drop4 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
    
    low_missing_features = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual',
                            'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt',
                            'MasVnrArea']
    df[low_missing_features] = df[low_missing_features].fillna('None')

    # Create Age Features BEFORE Dropping Year Columns
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # Combine all columns to drop
    cols_to_remove = columns_to_drop1 + columns_to_drop2 + columns_to_drop3 + columns_to_drop4

    # Drop columns
    df.drop(columns=[col for col in cols_to_remove if col in df.columns], inplace=True, errors='ignore')

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure test data has same features as training data
    if train_features is not None:
        # Identify missing columns in test set and add them all at once
        missing_cols = [col for col in train_features if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in train_features]

        # Add missing columns with default value 0 using pd.concat() to avoid fragmentation
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)

        # Remove any extra columns in test that were not in train
        df = df.drop(columns=extra_cols, errors='ignore')

        # Ensure same column order as train
        df = df[train_features]

    # Fill any remaining NaN values
    df.fillna(0, inplace=True)

    return df