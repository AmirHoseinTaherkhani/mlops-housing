import mlflow
import mlflow.sklearn
import optuna
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils import preprocess_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def objective(trial, X_train, y_train, X_val, y_val):
    # Hyperparameters to tune
    alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)  # Use suggest_float with log scale
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1)  # Only for ElasticNet

    # Create the model (ElasticNet, Lasso, or Ridge)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)  # You can replace this with Lasso or Ridge as needed

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_val_pred = model.predict(X_val)

    # Evaluate the model
    mse_val = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)

    # Select a sample input from the validation set for MLflow
    input_example = np.array(X_val.iloc[0]).reshape(1, -1)

    # Start a new MLflow run for each trial
    with mlflow.start_run(nested=True) as run:  # Nested run for each trial
        # Log hyperparameters and metrics with MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("mse_val", mse_val)
        mlflow.log_metric("r2_val", r2_val)
        
        # Log the trained model with an input example
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        # Store the MLflow run ID in the Optuna trial's attributes
        trial.set_user_attr("mlflow_run_id", run.info.run_id)

    return mse_val  # The objective function should return the value to minimize (MSE in this case)

def run_optuna_experiment(X_train, y_train, X_val, y_val):
    # Start the MLflow experiment
    mlflow.set_experiment("Hyperparameter Optimization with Optuna")

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")  # Minimize MSE
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)  # Run 50 trials

    # Print the best hyperparameters found
    print("Best trial:")
    print("  Value (MSE):", study.best_value)
    print("  Params:", study.best_params)

    # Retrieve the MLflow run ID from the best trial
    best_trial = study.best_trial
    best_run_id = best_trial.user_attrs["mlflow_run_id"]  # Now this will exist

    # Load the best model from MLflow
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

    # Generate predictions using the best model
    best_predictions = best_model.predict(X_val)

    # Calculate R² score
    r2_val = r2_score(y_val, best_predictions)
    print("Best R²:", r2_val)

    # Visualize residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val - best_predictions, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("True Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig("residual_plot.png", dpi=300)
    plt.show()

    # Log the residual plot
    mlflow.log_artifact("residual_plot.png")

    # Log the best model to MLflow again for convenience
    mlflow.sklearn.log_model(best_model, "best_model")

# Assuming data is preprocessed and split into X_train, y_train, X_val, y_val
# Load and preprocess the data
data = pd.read_csv("data/train.csv")

# Preprocess the data
data = preprocess_data(data, train_features=None, outlier_method="zscore")

# Split into features (X) and target (y)    
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=72)

# Run the Optuna experiment
run_optuna_experiment(X_train, y_train, X_val, y_val)
