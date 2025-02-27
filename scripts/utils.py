import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler

def remove_outliers_zscore(df, threshold=3):
    """Removes rows where any numerical column has a Z-score greater than the threshold."""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Select numerical features
    z_scores = stats.zscore(df[numerical_cols])
    filtered_df = df[(abs(z_scores) < threshold).all(axis=1)]
    
    print(f"Removed {df.shape[0] - filtered_df.shape[0]} outliers using Z-score method.")
    return filtered_df

def log_transform_target(df, column="SalePrice"):
    """Applies log transformation to the target variable."""
    df[column] = np.log1p(df[column])  # log1p prevents log(0) errors
    return df


def scale_features(df):
    """Scales numerical features using RobustScaler to reduce the impact of outliers."""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = RobustScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def remove_outliers_iqr(df):
    """Removes rows where any numerical column has values outside the IQR threshold."""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Select numerical features
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    print(f"Removed {df.shape[0] - filtered_df.shape[0]} outliers using IQR method.")
    return filtered_df


def preprocess_data(df, train_features=None, outlier_method="zscore"):
    """Applies preprocessing pipeline including outlier removal, feature encoding, and scaling."""

    # Handle missing values
    low_missing_features = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual',
                            'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt',
                            'MasVnrArea']
    df[low_missing_features] = df[low_missing_features].fillna('None')

    # Create Age Features BEFORE Dropping Year Columns
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # Drop irrelevant features
    features_to_drop = ['Id', 'LotFrontage', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'GarageArea',
                        'TotRmsAbvGrd', '1stFlrSF', 'YearBuilt', 'YearRemodAdd', 'YrSold',
                        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
    df.drop(columns=[col for col in features_to_drop if col in df.columns], inplace=True, errors='ignore')

    # Log-transform SalePrice to reduce skewness
    if "SalePrice" in df.columns:
        df["SalePrice"] = np.log1p(df["SalePrice"])

    # Remove Outliers
    if outlier_method == "zscore":
        df = remove_outliers_zscore(df)
    elif outlier_method == "iqr":
        df = remove_outliers_iqr(df)

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numerical features using RobustScaler
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = RobustScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Ensure test data has same features as training data
    if train_features is not None:
        missing_cols = [col for col in train_features if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in train_features]
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)
        df = df.drop(columns=extra_cols, errors='ignore')
        df = df[train_features]

    return df
