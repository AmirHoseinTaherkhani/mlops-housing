import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from experiment import run_optuna_experiment  # Import the function from experiment.py
from utils import preprocess_data

# Initialize MLflow tracking
mlflow.set_experiment("House Prices Prediction")
mlflow.set_tracking_uri("http://localhost:5001")  # Make sure MLflow tracking server is running

def run_experiment(train_path="data/train.csv", test_size=0.2, random_state=72):
    """Train and evaluate multiple models: Lasso, Ridge, ElasticNet with Optuna."""
    
    # Load and preprocess train data
    data = pd.read_csv(train_path)
    data = preprocess_data(data)

    # Split into features (X) and target (y)
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Run experiments with Optuna for hyperparameter optimization
    run_optuna_experiment(X_train, y_train, X_val, y_val)

def evaluate_on_test(test_path="data/test.csv", model_path="models/elastic_net_model.pkl", output_file="submission.csv"):
    """Evaluate the trained model on unseen test data and save predictions."""
    
    with mlflow.start_run():
        # Load the test data
        test_data = pd.read_csv(test_path)
        test_ids = test_data["Id"]

        # Load the trained model
        model = mlflow.sklearn.load_model(f"models/{model_path}")

        # Preprocess the test data
        test_data = preprocess_data(test_data, train_features=model.feature_names_in_)

        # Make predictions
        predictions = model.predict(test_data)

        # Save the predictions to a CSV file
        submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
        submission.to_csv(output_file, index=False)

        print(f"Predictions saved to {output_file}")

        # Log predictions file in MLflow
        mlflow.log_artifact(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate regression models for house price prediction.")
    parser.add_argument("--train", action="store_true", help="Train the models.")
    parser.add_argument("--test", action="store_true", help="Evaluate the model on test data.")
    parser.add_argument("--train_path", type=str, default="data/train.csv", help="Path to the train dataset CSV file.")
    parser.add_argument("--test_path", type=str, default="data/test.csv", help="Path to the test dataset CSV file.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of training data for validation split.")
    parser.add_argument("--random_state", type=int, default=72, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    if args.train:
        run_experiment(args.train_path, args.test_size, args.random_state)
    
    if args.test:
        evaluate_on_test(args.test_path)
