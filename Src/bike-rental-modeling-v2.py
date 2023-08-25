import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


def train_test_compare():
    # Read the CSV file into a pandas DataFrame
    csv_path = "../data/daily-bike-share.csv"
    df = pd.read_csv(csv_path)

    # Data preprocessing
    X = df.drop(columns=["instant", "dteday", "rentals"])
    categorical_columns = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    X[categorical_columns] = X[categorical_columns].astype("category")
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    y = df["rentals"]

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # List of regression models
    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree Regressor", DecisionTreeRegressor(random_state=42)),
        ("Random Forest Regressor", RandomForestRegressor(random_state=42)),
        ("Support Vector Regressor", SVR()),
    ]

    best_model = None
    best_mse = float("inf")

    # Train, evaluate, and compare models
    for name, model in models:
        print(f"Training and evaluating {name}")

        # Model training
        model.fit(X_train, y_train)

        # Model prediction
        y_pred = model.predict(X_test)

        # Model evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error:", mse)
        print("R-squared:", r2)
        print("-----------------------")

        # Select the best model based on MSE
        if mse < best_mse:
            best_mse = mse
            best_model = model

    print("Best Model:", best_model)
    print("Best MSE:", best_mse)
    return best_model


def hyperparameter_tuning(best_model):
    # Hyperparameters to tune
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        best_model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)

    # Print best parameters and scores
    print("Best Parameters:", grid_search.best_params_)
    print("Best MSE during Grid Search:", -grid_search.best_score_)

    return grid_search.best_estimator_


best_model = train_test_compare()
best_model_tuned = hyperparameter_tuning(best_model)
