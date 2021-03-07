import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from joblib import load


# Configure argument parser with sane defaults
parser = argparse.ArgumentParser(description="Calculate scoring metrics and return scores")
parser.add_argument("input_file", help="filename for input input csv containing scoring data")
parser.add_argument("-o", "--output_file", help="filename for output csv containing scored data", default="saved_scores.csv")
parser.add_argument("-m", "--model_file", help="saved serialized model object")
parser.add_argument("-d", "--decimals", help="decimal rounding for performance metrics", type=int, default=3)
args = parser.parse_args()


# Generate X, y data for model input and validation
def load_data(data: str) -> [pd.DataFrame, pd.Series]:
    X = pd.read_csv(data)
    y = X.pop("y")
    return X, y


# Use sklearn to calculate MSE then take the root
def calculate_rmse(actual: pd.Series, predicted: np.array) -> float:
    return np.sqrt(mean_squared_error(actual, predicted))


# Calculate percent accuracy using custom metric
def calculate_percent_accuracy(actual: pd.Series, predicted: np.array) -> float:
    return sum(abs(actual - predicted) <= 3.0) / len(actual)


# Predict using a trained model and then rescale using ECDFs
def generate_predictions(model: dict, data: pd.DataFrame) -> np.array:
    predictions = model["model"].predict(data)
    predictions = model["ecdf"](predictions)
    predictions = model["inverse_ecdf"](predictions)
    min_score = model["inverse_ecdf"].y.min()
    return np.nan_to_num(predictions, nan=min_score)


# Print RMSE and percent accuracy
def score_predictions(actual: pd.Series, predictions: np.array, decimals: int) -> None:
    rmse = calculate_rmse(actual, predictions)
    percent_accuracy = calculate_percent_accuracy(actual, predictions)
    print(f'RMSE: {rmse:.{decimals}f}')
    print(f'Percent Accuracy: {percent_accuracy:.{decimals}f}')
    return


# Saved predictions to csv
def save_csv(data: np.array, filename: str) -> None:
    np.savetxt(filename, data, delimiter=",")
    print(f"Saved data to {filename}")
    return


#
def score_data(model: dict, data: str) -> None:
    X, y = load_data(data)
    predictions = generate_predictions(model, X)
    score_predictions(y, predictions, args.decimals)
    save_csv(predictions, args.output_file)
    return


# If not specified use a default joblib dumped sklearn Pipeline
if args.model_file:
    model = load(args.model_file)
else:
    model = load("saved_model.joblib")

# Run scoring
score_data(model, args.input_file)
