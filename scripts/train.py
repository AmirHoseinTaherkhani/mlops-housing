import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
from utils import preprocess_data

# Initialize MLflow tracking
mlflow.set_experiment("House Prices Prediction")
# Set the MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5001")

def train_linear_regression(train_path="data/train.csv", test_size=0.2, random_state=72):
    """Train a Linear Regression model using train.csv (split into train/validation).
       Later, test it separately on test.csv."""
    
    np.random.seed(random_state)

    # Load and preprocess train data
    data = pd.read_csv(train_path)
    data = preprocess_data(data)

    # Split into train/validation sets
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Train model
    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        model = LinearRegression()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Evaluate model
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_val = mean_squared_error(y_val, y_val_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_val = r2_score(y_val, y_val_pred)

        # Log metrics
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mse_train", mse_train)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("mse_val", mse_val)
        mlflow.log_metric("r2_val", r2_val)

        # Ensure the model directory exists
        os.makedirs("models", exist_ok=True)

        # Save model
        model_path = "models/linear_regression.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained and saved: Validation Set MSE = {mse_val}, R² = {r2_val}")

    return model_path


def evaluate_on_test(test_path="data/test.csv", model_path="models/linear_regression.pkl", output_file="submission.csv"):
    """Evaluate the trained model on unseen test data and save predictions."""
    
    with mlflow.start_run():  # <-- Wrap everything in an MLflow run
        # Load test dataset
        test_data = pd.read_csv(test_path)
        test_ids = test_data["Id"]

        # Load trained model
        model = joblib.load(model_path)

        # Preprocess test data, ensuring it has the same features as training
        test_data = preprocess_data(test_data, train_features=model.feature_names_in_)

        # Predict SalePrice
        predictions = model.predict(test_data)

        # Save the submission file
        submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
        submission.to_csv(output_file, index=False)

        print(f"Predictions saved to {output_file}")

        # Log predictions file in MLflow
        mlflow.log_artifact(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Linear Regression model for house price prediction.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Evaluate the model on test data.")
    parser.add_argument("--train_path", type=str, default="data/train.csv", help="Path to the train dataset CSV file.")
    parser.add_argument("--test_path", type=str, default="data/test.csv", help="Path to the test dataset CSV file.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of training data for validation split.")
    parser.add_argument("--random_state", type=int, default=72, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    if args.train:
        train_linear_regression(args.train_path, args.test_size, args.random_state)
    
    if args.test:
        evaluate_on_test(args.test_path)