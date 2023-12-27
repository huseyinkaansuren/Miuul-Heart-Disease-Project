import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn.model_selection import (
    cross_validate,
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import joblib

pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 5000)

df_train = pd.read_csv("disease.csv")
df_train.head()

df_train.columns
df_train.info()


value_to_delete = 0
num_rows_to_delete = 210000

indices_to_delete = df_train[df_train["HeartDiseaseorAttack"] == value_to_delete].index
rows_to_delete = np.random.choice(
    indices_to_delete,
    size=min(num_rows_to_delete, len(indices_to_delete)),
    replace=False,
)

df_deleted_0 = df_train.loc[rows_to_delete]
df_deleted_0.to_csv("heart0.csv", index=False)

df_0 = df_train.drop(rows_to_delete)
df_0.shape

value_to_delete = 1
num_rows_to_delete = 1

indices_to_delete = df_train[df_train["HeartDiseaseorAttack"] == value_to_delete].index
rows_to_delete = np.random.choice(
    indices_to_delete,
    size=min(num_rows_to_delete, len(indices_to_delete)),
    replace=False,
)

df_deleted_1 = df_train.loc[rows_to_delete]
df_deleted_1.to_csv("heart1.csv")


df = df_0.drop(rows_to_delete)
df["HeartDiseaseorAttack"].value_counts()
df.shape

####### Visualize

# sns.set(style="whitegrid")

# plt.figure(figsize=(18, 12))

# for i, column in enumerate(df.columns):
#     plt.subplot(5, 5, i + 1)
#     sns.histplot(df[column], kde=True)
#     plt.title(column)

# plt.tight_layout()
# plt.savefig("distribution.png", bbox_inches="tight")
# plt.show()


# # Correlation
# corr_matrix = df.corr()

# plt.figure(figsize=(16, 12))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Korelasyon Matrisi")
# plt.savefig("correlation.png", bbox_inches="tight")
# plt.show()


# # Boxplot

# plt.figure(figsize=(18, 12))

# for i, column in enumerate(df.columns):
#     plt.subplot(5, 5, i + 1)
#     sns.boxplot(x="Sex", y=column, data=df)
#     plt.title(column)

# plt.tight_layout()
# plt.savefig("boxplot.png", bbox_inches="tight")
# plt.show()


# df.describe().T

# # Categorical

# plt.figure(figsize=(18, 12))

# for i, column in enumerate(df.columns):
#     if len(df[column].unique()) < 10:  # Kategorik değişkenler için
#         plt.subplot(5, 5, i + 1)
#         sns.countplot(x=column, data=df)
#         plt.title(column)

# plt.tight_layout()
# plt.savefig("categorical.png", bbox_inches="tight")
# plt.show()


def check_df(dataframe, head=5, tail=5):
    """
    Display various information about the DataFrame, including shape, data types, head, tail, missing values, and quantiles.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        head : int, optional
            Number of rows to display for the head. Default is 5.
        tail : int, optional
            Number of rows to display for the tail. Default is 5.

    Returns
    ------
        None

    Prints the following information:
    - Shape of the DataFrame.
    - Info about the DataFrame, including data types, non-null counts, and memory usage.
    - First 'head' rows of the DataFrame.
    - Last 'head' rows of the DataFrame.
    - Number of missing values for each column.
    - Quantiles (0%, 5%, 50%, 95%, 99%, 100%) for numerical columns with at least 10 unique values.

    Examples
    ------
        # Display information for the entire DataFrame (default head=5, tail=5)
        check_df(df)

        # Display information with a custom number of rows for head and tail
        check_df(df, head=10, tail=15)
    """

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.info())
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    numerical_cols = [
        col
        for col in dataframe.columns
        if dataframe[col].dtype in ["int64", "float64"]
        and dataframe[col].nunique() >= 10
    ]
    print(dataframe[numerical_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identify and categorize columns in a DataFrame based on their data types and cardinality.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        cat_th : int, optional
            Numerical threshold for considering a numerical variable as categorical. Default is 10.
        car_th : int, optional
            Categorical threshold for considering a categorical variable as cardinal. Default is 20.

    Returns
    ------
        cat_cols : list
            List of categorical variables.
        num_cols : list
            List of numerical variables.
        cat_but_car : list
            List of categorical variables with cardinality exceeding 'car_th'.
        num_but_cat : list
            List of numerical variables with cardinality below 'cat_th'.

    Examples
    ------
        # Example: Categorize and display column information for a DataFrame
        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=8, car_th=15)
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    """
    Calculate the lower and upper bounds for identifying outliers in a numerical column.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        col_name : str
            The name of the numerical column for which outlier thresholds will be calculated.
        q1 : float, optional
            The lower quartile value. Default is 0.10.
        q3 : float, optional
            The upper quartile value. Default is 0.90.

    Returns
    ------
        low_limit : float
            The lower threshold for identifying outliers.
        up_limit : float
            The upper threshold for identifying outliers.

    Examples
    ------
        # Example: Calculate outlier thresholds for a numerical column
        low_limit, up_limit = outlier_thresholds(df, 'numeric_column')
    """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    """
    Check for outliers in a numerical column of a DataFrame based on custom quantiles.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        col_name : str
            The name of the numerical column to check for outliers.
        q1 : float, optional
            The lower quantile value for calculating the lower threshold. Default is 0.10.
        q3 : float, optional
            The upper quantile value for calculating the upper threshold. Default is 0.90.

    Returns
    ------
        str
            A string indicating the presence of outliers in the specified numerical column.

    Examples
    ------
        # Example: Check for outliers in a numerical column with custom quantiles
        result = check_outlier(df, 'numeric_column', q1=0.05, q3=0.95)
        print(result)

        # Example 2: Check for outliers for multiple numerical columns
        for col in num_cols:
            print(check_outlier(df, col))
    """

    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    print(f"{col_name} Lower Limit: {low_limit}, {col_name} Upper Limit: {up_limit}")

    outliers = dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ]

    num_outliers = outliers.shape[0]  # Count the number of outliers

    if num_outliers > 0:
        return f"{col_name} : {num_outliers} : True"
    else:
        return f"{col_name} : {num_outliers} : False"


for col in num_cols:
    check_outlier(df, col)


def replace_with_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    """
    Replace values in a numerical column with their respective outlier thresholds based on custom quantiles.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        variable : str
            The name of the numerical column for which outliers will be replaced with their thresholds.
        q1 : float, optional
            The lower quantile value for calculating the lower threshold. Default is 0.10.
        q3 : float, optional
            The upper quantile value for calculating the upper threshold. Default is 0.90.

    Returns
    ------
        None

    Modifies the DataFrame in-place by replacing values below the lower threshold with the lower threshold,
    and values above the upper threshold with the upper threshold.

    Examples
    ------
        # Example: Replace outliers in a numerical column with custom quantiles
        replace_with_thresholds(df, 'numeric_column', q1=0.05, q3=0.95)

        # Example 2: Replace outliers for multiple numerical columns
        for col in num_cols:
            replace_with_thresholds(df, col)
    """

    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)

    # Explicitly cast the limits to the data type of the column
    low_limit = low_limit.astype(dataframe[variable].dtype)
    up_limit = up_limit.astype(dataframe[variable].dtype)

    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "BMI")


def target_summary_with_num(dataframe, target, num_col):
    """
    Generate a summary for a numerical column grouped by a target variable in a DataFrame.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        target : str
            The name of the target variable by which the numerical column will be grouped.
        num_col : str
            The name of the numerical column for which the summary is generated.

    Returns
    ------
        None

    Prints the mean of the specified numerical column grouped by the target variable.

    Examples
    ------
        # Example 1: Summary for a numerical column grouped by a binary target variable
        target_summary_with_num(df, 'target_variable', 'numeric_column')

        # Example 2: Summary for multiple numerical columns
        for col in num_cols:
            target_summary_with_num(df, 'target_variable', col)
    """

    print(dataframe.groupby(target).agg({num_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, cat_col):
    """
    Generate a summary for a categorical column in relation to a target variable in a DataFrame.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        target : str
            The name of the target variable.
        cat_col : str
            The name of the categorical column for which the summary is generated.

    Returns
    ------
        None

    Prints a DataFrame with the mean of the target variable for each category in the specified categorical column.

    Examples
    ------
        # Example 1: Summary for a categorical column in relation to a binary target variable
        target_summary_with_cat(df, 'target_variable', 'categorical_column')

        # Example 2: Summary for multiple categorical columns
        for col in cat_cols:
            target_summary_with_cat(df, 'target_variable', col)
    """

    print(
        pd.DataFrame(
            {
                f"{target}_MEAN": dataframe.groupby(cat_col, observed=False)[
                    target
                ].mean()
            }
        ),
        end="\n\n\n",
    )
    print("#######################################", end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "HeartDiseaseorAttack", col)

df["Income"].value_counts()

df["Income"]
len(df.columns)

# Feature Engineering

df.head()

araliklar = [18, 25, 30, 35, df["BMI"].max()]
df["BMI_cat"] = pd.cut(
    df["BMI"],
    bins=araliklar,
    labels=["Underweight", "Normalweight", "Overweight", "Obese"],
)


df["BMI"].describe()


df.groupby("Income").agg({"NoDocbcCost": "mean"})

df.loc[(df["Income"] <= 4), "Income_Cat"] = "Poor"
df.loc[(df["Income"] >= 5) & (df["Income"] < 8), "Income_Cat"] = "Normal"
df.loc[(df["Income"] >= 8), "Income_Cat"] = "Rich"

df["HealthyLifestyleScore"] = (
    df["PhysActivity"]
    + df["Fruits"]
    + df["Veggies"]
    - df["Smoker"]
    - df["HvyAlcoholConsump"]
)

risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
df["RiskFactorCount"] = df[risk_factors].sum(axis=1)
df["OverallHealthScore"] = df["GenHlth"] + df["MentHlth"]


bins = [0, 2, 4, 6, 8, 10, 12]
labels = ["Low", "Lower-Mid", "Mid", "Upper-Mid", "High", "Very High"]
df["SES"] = pd.cut(
    df["Education"] + df["Income"], bins=bins, labels=labels, right=False
)
# Sosyoekonomik statü


df["UnhealthyLifestyle"] = (
    df[["Smoker", "HvyAlcoholConsump", "NoDocbcCost"]].any(axis=1).astype(int)
)

df["DietaryHabitsScore"] = df["Fruits"] + df["Veggies"]
len(df.columns)

df.head()

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


# Encoding
# df_model_train = df.copy()
df_model_train = df_train.copy()


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first, dtype="int"
    )
    return dataframe


df_model_train = one_hot_encoder(df, cat_cols, True)


# MODEL
X = df_model_train.drop("HeartDiseaseorAttack_1.0", axis=1)
y = df_model_train["HeartDiseaseorAttack_1.0"]

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# from sklearn.preprocessing import MinMaxScaler


# sc = MinMaxScaler((0, 1))
# X[num_cols] = sc.fit_transform(X[num_cols])


# X_scaled = sc.fit_transform(X)


classifiers = [
    ("LR", LogisticRegression(max_iter=1000)),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("RF", RandomForestClassifier()),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
    ("LightGBM", LGBMClassifier(force_row_wise=True, verbose=-1)),
]


def evaluate_models(models, X, y, scoring="roc_auc", test_size=0.2, random_state=None):
    """
    Evaluate a list of models using cross-validation and provide accuracy scores on a separate test set.

    Parameters:
    - models: List of tuples containing (name, model_instance)
    - X: Feature matrix
    - y: Target variable
    - scoring: Scoring metric for cross-validation
    - test_size: Percentage of data to use as a test set
    - random_state: Random seed for reproducibility

    Returns:
    - model_performance: Dictionary containing model names and their cross-validation scores
    - test_scores: Dictionary containing model names and their accuracy scores on the test set
    """

    print("Evaluating Base Models...", end="\n")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Set random seed for cross-validation
    if random_state:
        np.random.seed(random_state)

    # Make a dictionary to keep model scores
    model_performance = {}
    test_scores = {}

    for name, model in models:
        # Cross-validation
        cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
        model_performance[name] = round(cv_results["test_score"].mean(), 4)

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Evaluate on the test set
        y_pred = model.predict(X_test)
        test_scores[name] = accuracy_score(y_test, y_pred)

        # Print results
        print(
            f"{scoring}: {model_performance[name]} (CV) | Accuracy: {test_scores[name]} (Test) - {name}"
        )

    return model_performance, test_scores


evaluate_models(classifiers, X, y)

import os

# Set LOKY_MAX_CPU_COUNT to silence the warning about physical cores
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


import warnings

from sklearn.exceptions import ConvergenceWarning

# Suppress specific Logistic Regression warnings
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.linear_model._logistic"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.linear_model._logistic"
)
warnings.filterwarnings(
    "ignore", category=ConvergenceWarning, module="sklearn.linear_model._sag"
)

warnings.filterwarnings(
    "ignore", category=ConvergenceWarning, module="sklearn.linear_model._sag"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.linear_model._logistic"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.linear_model._logistic"
)

# Your code follows...


# Suppress specific SciPy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

# Your code follows...


def hyperparameter_optimization_random_search(
    X, y, classifiers, cv=3, n_iter=10, scoring=None, verbose=False, random_state=None
):
    """
    Perform hyperparameter optimization for multiple classifiers using RandomizedSearchCV.

    Parameters:
    -----------
    X : array-like or pd.DataFrame
        Feature matrix.
    y : array-like
        Target variable.
    classifiers : list of tuples
        List containing tuples of (classifier_name, classifier_instance, param_distributions).
    cv : int, optional (default=3)
        Number of cross-validation folds.
    scoring : str or list, optional
        Scoring metric(s) to evaluate during optimization. If None, uses default metrics:
        ["accuracy", "f1", "roc_auc"].
    verbose : bool, optional (default=False)
        If True, display detailed information during randomized search.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary containing the best models for each classifier.

    Examples:
    ---------
    # Example: Perform hyperparameter optimization for random forest and logistic regression
    classifiers = [
        ("RandomForest", RandomForestClassifier(), {"n_estimators": [10, 50, 100]}),
        ("LogisticRegression", LogisticRegression(), {"C": [0.1, 1, 10]})
    ]
    best_models = hyperparameter_optimization(X_train, y_train, classifiers)
    """

    print("Hyperparameter Optimization....")
    best_models = {}

    # Set random seed for reproducibility
    if random_state:
        np.random.seed(random_state)

    for name, classifier, param_distributions in classifiers:
        print(f"########## {name} ##########")

        st = time.time()

        # If scoring is not specified, use all available scoring metrics
        if scoring is None:
            scoring = ["accuracy", "f1", "roc_auc"]

        # Convert single string scoring to a list
        if isinstance(scoring, str):
            scoring = [scoring]

        # Loop through each scoring metric
        for single_scoring in scoring:
            print(f"Scoring: {single_scoring}")

            # Before optimization
            cv_results_before = cross_validate(
                classifier, X, y, cv=cv, scoring=single_scoring
            )
            print(
                f"{single_scoring} (Before): Mean={cv_results_before['test_score'].mean():.4f}, Std={cv_results_before['test_score'].std():.4f}"
            )

            # Randomized search for hyperparameter optimization
            rs_best = RandomizedSearchCV(
                classifier,
                param_distributions,
                n_iter=n_iter,
                cv=cv,
                n_jobs=-1,
                verbose=verbose,
                scoring=single_scoring,
                random_state=random_state,
            ).fit(X, y)

            # After optimization
            cv_results_after = cross_validate(
                rs_best.best_estimator_, X, y, cv=cv, scoring=single_scoring
            )
            print(
                f"{single_scoring} (After): Mean={cv_results_after['test_score'].mean():.4f}, Std={cv_results_after['test_score'].std():.4f}",
                end="\n\n",
            )

        print(f"{name} best params: {rs_best.best_params_}", end="\n")
        print(f"It took {time.time() - st} seconds", end="\n\n")
        best_models[name] = rs_best.best_estimator_

    return best_models


rf_params = {
    "n_estimators": [10, 50, 100, 200, 300, 400, 500],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "max_features": ["auto", "sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "max_samples": [None, 0.5, 0.7, 0.9, 1.0],
    "random_state": [None, 42, 123, 789],
}

import warnings

# Suppress specific Logistic Regression warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

# Your code follows...
from scipy.stats import uniform, randint

lr_params = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": uniform(0.1, 10.0),
    "fit_intercept": [True, False],
    "max_iter": randint(50, 200),
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "verbose": [0],
}


lightgbm_params = {
    "objective": ["binary"],  # Change the objective to binary classification
    "metric": ["binary_error"],  # Change the metric to binary classification
    "boosting_type": ["gbdt"],
    "num_leaves": [31, 40],
    "learning_rate": [0.01, 0.05, 0.1],
    "feature_fraction": np.linspace(0.7, 1.0, 4),
    "bagging_fraction": np.linspace(0.7, 1.0, 4),
    "bagging_freq": [5, 10],
    "verbosity": [-1],
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": np.linspace(0.8, 1.0, 4),
    "colsample_bytree": np.linspace(0.8, 1.0, 4),
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.0, 0.1, 0.5],
    "verbosity": [-1],
}


xgboost_params = {
    "learning_rate": uniform(0.01, 0.1),
    "n_estimators": randint(100, 201),
    "max_depth": randint(3, 8),
    "min_child_weight": randint(1, 4),
    "gamma": uniform(0, 0.2),
    "subsample": uniform(0.8, 0.2),
    "colsample_bytree": uniform(0.8, 0.2),
    "objective": ["binary:logistic"],  # Change the objective to binary classification
    "eval_metric": [
        "logloss"
    ],  # Optionally, you can specify the evaluation metric for binary classification
    "n_jobs": [-1],
}


# catboost_params_multiclass = {
#     "iterations": randint(100, 201),  # Random integer between 100 and 200
#     "learning_rate": uniform(0.01, 0.1),  # Random float between 0.01 and 0.1
#     "depth": randint(6, 11),  # Random integer between 6 and 10
#     "l2_leaf_reg": randint(3, 8),  # Random integer between 3 and 7
#     "border_count": [32, 64, 128],
#     "loss_function": ["MultiClass"],
#     "verbose": [0],
# }


hyperparam_classifiers = [
    # ("SVC", SVC(), svc_params)("RF", RandomForestClassifier(), rf_params),
    # ("LR", LogisticRegression(max_iter=2000), lr_params),
    ("LightGBM", LGBMClassifier(verbose=-1), lightgbm_params),
    (
        "XGBoost",
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        xgboost_params,
    ),
    # ("CatBoost", CatBoostClassifier(verbose=False), catboost_params_multiclass),
]

import time

hyper_params = hyperparameter_optimization_random_search(
    X,
    y,
    hyperparam_classifiers,
    cv=3,
    n_iter=100,
    scoring="accuracy",
)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

df = pd.read_csv("new_train30k.csv")


def preprocess_data(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_processed = df.copy()

    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df_processed["RiskFactorCount"] = df_processed[risk_factors].sum(axis=1)
    df_processed["OverallHealthScore"] = (
        df_processed["GenHlth"] + df_processed["MentHlth"]
    )
    cat_cols, num_cols, cat_but_car, cat_but_num = grab_col_names(df_processed)

    X = df_processed.drop("HeartDiseaseorAttack", axis=1)
    y = df_processed["HeartDiseaseorAttack"]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y


X, y = preprocess_data(df)


def voting_classifier(models, X, y, cv=3, scoring=None, voting="soft"):
    """
    Create a Voting Classifier using the specified models and evaluate its performance.

    Parameters:
    -----------
    models : dict
        Dictionary containing (model_name, model_instance) pairs.
    X : array-like or pd.DataFrame
        Feature matrix.
    y : array-like
        Target variable.
    cv : int, optional (default=3)
        Number of cross-validation folds.
    scoring : str or list, optional
        Scoring metric(s) to evaluate during cross-validation.
        If None, uses default metrics: ["accuracy", "f1", "roc_auc"].
    voting : {'hard', 'soft'}, optional (default='soft')
        Type of voting strategy for the ensemble.

    Returns:
    --------
    VotingClassifier
        Trained Voting Classifier model.

    Examples:
    ---------
    # Example: Create a Voting Classifier using Logistic Regression, Random Forest, and SVM
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }
    voting_clf = voting_classifier(models, X_train, y_train)
    """
    print("Voting Classifier...", end="\n\n")

    if not scoring:
        scoring = ["accuracy", "f1", "roc_auc"]

    estimators = [(name, model) for name, model in models.items()]

    voting_clf = VotingClassifier(estimators=estimators, voting=voting).fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=cv, scoring=scoring)

    for score in scoring:
        print(f"{score.capitalize()}: {cv_results[f'test_{score}'].mean()}")

    return voting_clf


lgbm_best_params = {
    "verbosity": -1,
    "subsample": 1.0,
    "reg_lambda": 0.1,
    "reg_alpha": 0.1,
    "objective": "binary",
    "num_leaves": 31,
    "n_estimators": 100,
    "min_child_weight": 5,
    "metric": "binary_error",
    "max_depth": 7,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "colsample_bytree": 0.9333333333333333,
    "boosting_type": "gbdt",
    "bagging_freq": 5,
    "bagging_fraction": 1.0,
}

xgb_best_params = {
    "colsample_bytree": 0.980275617338626,
    "eval_metric": "logloss",
    "gamma": 0.1254870784084415,
    "learning_rate": 0.03548371005683994,
    "max_depth": 5,
    "min_child_weight": 3,
    "n_estimators": 182,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "subsample": 0.8270677823080291,
}
votingmodels = {
    # "CB": CatBoostClassifier(),
    # "LR": LogisticRegression(max_iter=1000),
    # "RF": RandomForestClassifier(),*
    "LightGBM": LGBMClassifier(**lgbm_best_params),
    "XGBoost": XGBClassifier(**xgb_best_params),
}


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=52
)

voting_clf = voting_classifier(votingmodels, X_train, y_train, 3)


vt_lcf = voting_clf.fit(X, y)
y_pred = vt_lcf.predict(X)

from sklearn.metrics import classification_report

# Generate and print the classification report
print("Classification Report on New Data:")
print(classification_report(y, y_pred))
joblib.dump(vt_lcf, "final_final_model.joblib")

df_test = pd.read_csv("testData20k.csv")


def preprocess_data(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_processed = df.copy()

    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df_processed["RiskFactorCount"] = df_processed[risk_factors].sum(axis=1)
    df_processed["OverallHealthScore"] = (
        df_processed["GenHlth"] + df_processed["MentHlth"]
    )
    cat_cols, num_cols, cat_but_car, cat_but_num = grab_col_names(df_processed)

    X = df_processed.drop("HeartDiseaseorAttack", axis=1)
    y = df_processed["HeartDiseaseorAttack"]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y


X, y = preprocess_data(df_test)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features.columns}
    )
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title("Features")
    plt.tight_layout()
    if save:
        plt.savefig("importances.png")
    plt.show()


plot_importance(voting_clf, X_train, save=False)


import joblib

joblib.dump(vt_lcf, "final_model.joblib")
joblib.dump(voting_clf, "voting_clf.joblib")

df_test = pd.read_csv("heart0.csv")

risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
df_test["RiskFactorCount"] = df_test[risk_factors].sum(axis=1)
df_test["OverallHealthScore"] = df_test["GenHlth"] + df_test["MentHlth"]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first, dtype="int"
    )
    return dataframe


df_model_train = one_hot_encoder(df, cat_cols, True)


X = df_model_train.drop("HeartDiseaseorAttack_1.0", axis=1)
y = df_model_train["HeartDiseaseorAttack_1.0"]

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])


import pandas as pd
from sklearn.preprocessing import StandardScaler


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first, dtype="int"
    )
    return dataframe


ne_test_df = one_hot_encoder(df_test, cat_cols, True)


def preprocess_data(df, cat_cols):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_processed = df.copy()

    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df_processed["RiskFactorCount"] = df_processed[risk_factors].sum(axis=1)
    df_processed["OverallHealthScore"] = (
        df_processed["GenHlth"] + df_processed["MentHlth"]
    )

    df_encoded = pd.get_dummies(
        df_processed, columns=cat_cols, drop_first=True, dtype="int"
    )

    X = df_encoded.drop("HeartDiseaseorAttack_1.0", axis=1)
    y = df_encoded["HeartDiseaseorAttack_1.0"]

    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y


cat_cols, num_cols, cat_but_car, cat_but_num = grab_col_names(df_test)
X, y = preprocess_data(df_test, cat_cols)
# Usage
X_train, y_train = preprocess_data(df_train)  # Replace df_train with your training data
X_test, y_test = preprocess_data(df_test)  # Replace df_test with your testing data


df_test = pd.read_csv("heart0.csv")
replace_with_thresholds(df_test, "BMI")
df_test.columns
df_test.info()

import pandas as pd

df_deneme = pd.read_csv("heart1.csv")


def preprocess_data(df):
    # df_processed = df.copy()
    cat_cols, num_cols, cat_but_car, cat_but_num = grab_col_names(df)

    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df["RiskFactorCount"] = df[risk_factors].sum(axis=1)
    df["OverallHealthScore"] = df["GenHlth"] + df["MentHlth"]

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype="int")
    df_encoded.head()

    X = df_encoded.drop("HeartDiseaseorAttack_1.0", axis=1)
    y = df_encoded["HeartDiseaseorAttack_1.0"]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y


df_test = pd.read_csv("disease.csv")

X, y = preprocess_data(df_test)


def preprocess_data(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_processed = df.copy()

    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df_processed["RiskFactorCount"] = df_processed[risk_factors].sum(axis=1)
    df_processed["OverallHealthScore"] = (
        df_processed["GenHlth"] + df_processed["MentHlth"]
    )
    cat_cols, num_cols, cat_but_car, cat_but_num = grab_col_names(df_processed)

    cat_cols.remove("HeartDiseaseorAttack")
    df_encoded = pd.get_dummies(
        df_processed, columns=cat_cols, drop_first=True, dtype="int"
    )

    X = df_encoded.drop("HeartDiseaseorAttack", axis=1)
    y = df_encoded["HeartDiseaseorAttack"]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y


X, y = preprocess_data(df_test)


from sklearn.metrics import classification_report


loaded_model = joblib.load("logistic_regression_model.joblib")
loaded_model = joblib.load("voting_clf.joblib")
y_pred = loaded_model.predict(X)

# Generate a classification report
report = classification_report(y, y_pred)

# Print the classification report
print("Classification Report:\n", report)

df_train = pd.read_csv("disease.csv")

df_train = pd.read_csv("disease.csv")
value_to_delete = 0
num_rows_to_delete = 220000

indices_to_delete = df_train[df_train["HeartDiseaseorAttack"] == value_to_delete].index
rows_to_delete = np.random.choice(
    indices_to_delete,
    size=min(num_rows_to_delete, len(indices_to_delete)),
    replace=False,
)

df_deleted_0 = df_train.loc[rows_to_delete]
df_deleted_0.to_csv(
    "heart0.csv", index=False
)  # Set index=False to exclude the index column

df_0 = df_train.drop(rows_to_delete)


value_to_delete = 1
num_rows_to_delete = 12000

indices_to_delete = df_train[df_train["HeartDiseaseorAttack"] == value_to_delete].index
rows_to_delete = np.random.choice(
    indices_to_delete,
    size=min(num_rows_to_delete, len(indices_to_delete)),
    replace=False,
)

df_deleted_1 = df_train.loc[rows_to_delete]
df_deleted_1.to_csv("heart1.csv", index=False)

df = df_0.drop(rows_to_delete)

df.to_csv("disease_cleaned.csv", index=False)


df = pd.read_csv("heart1.csv")


def preprocess_data(df):
    # df_processed = df.copy()
    cat_cols, num_cols, cat_but_car, cat_but_num = grab_col_names(df)

    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df["RiskFactorCount"] = df[risk_factors].sum(axis=1)
    df["OverallHealthScore"] = df["GenHlth"] + df["MentHlth"]

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype="int")
    print(df_encoded.head())
    print(df_encoded.columns)

    # X = df_encoded.drop("HeartDiseaseorAttack_1.0", axis=1)
    # y = df_encoded["HeartDiseaseorAttack_1.0"]

    # scaler = StandardScaler()
    # X[num_cols] = scaler.fit_transform(X[num_cols])

    # return X, y


df = pd.read_csv("heart1.csv")
df2 = pd.read_csv("heart0.csv")
df_half = df2.sample(frac=0.05, random_state=42)
df1 = pd.concat([df, df_half])

df1.to_csv("test.csv", index=False)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identify and categorize columns in a DataFrame based on their data types and cardinality.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        cat_th : int, optional
            Numerical threshold for considering a numerical variable as categorical. Default is 10.
        car_th : int, optional
            Categorical threshold for considering a categorical variable as cardinal. Default is 20.

    Returns
    ------
        cat_cols : list
            List of categorical variables.
        num_cols : list
            List of numerical variables.
        cat_but_car : list
            List of categorical variables with cardinality exceeding 'car_th'.
        num_but_cat : list
            List of numerical variables with cardinality below 'cat_th'.

    Examples
    ------
        # Example: Categorize and display column information for a DataFrame
        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=8, car_th=15)
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car, num_but_cat


def preprocess_data(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_processed = df.copy()

    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df_processed["RiskFactorCount"] = df_processed[risk_factors].sum(axis=1)
    df_processed["OverallHealthScore"] = (
        df_processed["GenHlth"] + df_processed["MentHlth"]
    )
    cat_cols, num_cols, cat_but_car, cat_but_num = grab_col_names(df_processed)

    cat_cols.remove("HeartDiseaseorAttack")
    df_encoded = pd.get_dummies(
        df_processed, columns=cat_cols, drop_first=True, dtype="int"
    )

    X = df_encoded.drop("HeartDiseaseorAttack", axis=1)
    y = df_encoded["HeartDiseaseorAttack"]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y


X, y = preprocess_data(df1)

loaded_model = joblib.load("final_model.joblib")
loaded_model = joblib.load("voting_clf.joblib")
y_pred = loaded_model.predict(X)

# Generate a classification report
report = classification_report(y, y_pred)

# Print the classification report
print("Classification Report:\n", report)
df_half["HeartDiseaseorAttack"].value_counts()
df_half.columns

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")
df.drop("HeartDiseaseorAttack", inplace=True)


sex_mapping = {1: 'Male', : 'Female'}

# Convert numerical representation to words
df['Sex'] = df['Sex'].map(sex_mapping)

# Count the frequencies of each category
sex_counts = df['Sex'].value_counts().to_dict()

# Create WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(sex_counts)

# Display the WordCloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()