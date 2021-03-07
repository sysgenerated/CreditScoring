import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.distributions import ECDF, monotone_fn_inverter
from joblib import dump


# model = XGBRegressor(random_state=2021)
model = XGBRegressor(random_state=2021, n_estimators=900, max_depth=9, learning_rate=0.1, subsample=0.4)
# model = LGBMRegressor(random_state=2021)
# model = LGBMRegressor(random_state=2021, n_estimators=900, max_depth=9, learning_rate=0.1, subsample=0.4)

grid_search = False
save_model = True

param_grid = {
    'n_estimators': [300, 500, 700, 900],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [5, 7, 9],
    'subsample': [0.4, 0.5, 1.0]}


def load_data(data):
    X = pd.read_csv(data)
    y = X.pop("y")
    return X, y


def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def calculate_percent_accuracy(actual, predicted):
    return sum(abs(actual - predicted) <= 3.0) / len(actual)


def generate_scalers(model, X_train, y_train):
    predictions_train = model.predict(X_train)
    actual_ecdf = ECDF(y_train)
    actual_ecdf_inv = monotone_fn_inverter(actual_ecdf, y_train)
    actual_ecdf_inv.bounds_error = False
    prediction_ecdf = ECDF(predictions_train)
    return prediction_ecdf, actual_ecdf_inv


def generate_finalized_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    prediction_ecdf, actual_ecdf_inv = generate_scalers(model, X_train, y_train)
    return {"model": model, "ecdf": prediction_ecdf, "inverse_ecdf": actual_ecdf_inv}


def generate_scaled_predictions(finalized_model, data):
    predictions = finalized_model["model"].predict(data)
    predictions = finalized_model["ecdf"](predictions)
    predictions = finalized_model["inverse_ecdf"](predictions)
    min_score = finalized_model["inverse_ecdf"].y.min()
    return np.nan_to_num(predictions, nan=min_score)


def print_summary(actual, predictions, identifier):
    print("model " + identifier + " rmse: %.3f" % calculate_rmse(actual, predictions))
    print("model " + identifier + " acc: %.3f" % calculate_percent_accuracy(actual, predictions))


def save_csv(data: np.array, filename: str) -> None:
    np.savetxt(filename, data, delimiter=",")
    print(f"Saved data to {filename}")
    return


X,y = load_data("./input_data.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Create sklearn Pipelines for numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=999999)),
    ('scaler', StandardScaler())])

# Define numerical features by dtype, in this dataset this is all columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Create sklearn ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)])

# Create model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', model)])

if grid_search:
    model = GridSearchCV(model, param_grid, n_jobs=4)

finalized_model = generate_finalized_model(model, X_train, y_train)

if grid_search:
    print(model.best_params_)


predictions_train = generate_scaled_predictions(finalized_model, X_train)
predictions_test = generate_scaled_predictions(finalized_model, X_test)
print_summary(y_train, predictions_train, "train")
print_summary(y_test, predictions_test, "test")

if save_model:
    finalized_model = generate_finalized_model(model, X, y)
    predictions_final = generate_scaled_predictions(finalized_model, X)
    print_summary(y, predictions_final, "final")
    save_csv(predictions_final, "final_moodel_predictions.csv")
    dump(finalized_model, "saved_model.joblib", compress=True)
